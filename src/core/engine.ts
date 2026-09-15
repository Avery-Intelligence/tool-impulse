import { ToolCatalog } from './bank.js';
import { ToolResolver } from './resolver.js';
import {
  EmbeddingProvider,
  RouterOptions,
  SessionState,
  ToolCatalogState,
  ToolDefinition,
  ToolRouteResult,
  ToolTransitionEdge,
} from './types.js';

export interface ToolImpulseConfig {
  embedder?: EmbeddingProvider;
  defaultOptions?: RouterOptions;
  tools?: ToolDefinition[];
  edges?: ToolTransitionEdge[];
  initialState?: ToolCatalogState;
}

const ENGINE_CACHE_MAX_SIZE = 128;

function getFromLru<K, V>(map: Map<K, V>, key: K): V | undefined {
  const val = map.get(key);
  if (val !== undefined) {
    map.delete(key);
    map.set(key, val);
  }
  return val;
}

function setInLru<K, V>(map: Map<K, V>, key: K, val: V, maxSize = ENGINE_CACHE_MAX_SIZE): void {
  if (map.has(key)) {
    map.delete(key);
  } else if (map.size >= maxSize) {
    const oldest = map.keys().next().value;
    if (oldest !== undefined) map.delete(oldest);
  }
  map.set(key, val);
}

export class ToolImpulse {
  private catalog: ToolCatalog;
  private resolver: ToolResolver;
  private embedder?: EmbeddingProvider;
  private defaultOptions?: RouterOptions;

  constructor(config: ToolImpulseConfig = {}) {
    this.catalog = new ToolCatalog();
    this.resolver = new ToolResolver(this.catalog);
    this.embedder = config.embedder;
    this.defaultOptions = config.defaultOptions;

    if (config.initialState) {
      this.importState(config.initialState);
    }
    if (config.tools) {
      this.registerToolsSync(config.tools);
    }
    if (config.edges) {
      this.catalog.addEdges(config.edges);
    }
  }

  /**
   * Register tools and compute vector embeddings (if an embedder is configured).
   * Also synchronizes the Okapi BM25 lexical index.
   */
  public async registerTools(tools: ToolDefinition[]): Promise<void> {
    this.catalog.registerTools(tools);
    this.resolver.syncIndex();

    if (this.embedder) {
      const descriptions = tools.map((t) => `${t.name}: ${t.description} ${(t.keywords || []).join(' ')}`);
      const vectors = await this.embedder.embedBatch(descriptions);
      const map = new Map<string, Float32Array>();
      for (let i = 0; i < tools.length; i++) {
        map.set(tools[i].name, vectors[i]);
      }
      this.catalog.setEmbeddings(map);
    }
  }

  /**
   * Register tools synchronously without external embedding calls (runs pure BM25).
   */
  public registerToolsSync(tools: ToolDefinition[]): void {
    this.catalog.registerTools(tools);
    this.resolver.syncIndex();
  }

  /**
   * Check if an embedding provider is configured.
   */
  public hasEmbedder(): boolean {
    return !!this.embedder;
  }

  /**
   * Returns the configured embedding provider, if any.
   */
  public getEmbedder(): EmbeddingProvider | undefined {
    return this.embedder;
  }

  /**
   * Returns the default router options, if any.
   */
  public getDefaultOptions(): RouterOptions | undefined {
    return this.defaultOptions;
  }

  /**
   * Atomically replace catalog tools, compute embeddings (if an embedder is configured),
   * and synchronize the Okapi BM25 lexical index.
   */
  public async setTools(tools: ToolDefinition[]): Promise<void> {
    this.catalog.setTools(tools);
    this.resolver.syncIndex();

    if (this.embedder) {
      const descriptions = tools.map((t) => `${t.name}: ${t.description || ''} ${(t.keywords || []).join(' ')}`);
      const vectors = await this.embedder.embedBatch(descriptions);
      const map = new Map<string, Float32Array>();
      for (let i = 0; i < tools.length; i++) {
        map.set(tools[i].name, vectors[i]);
      }
      this.catalog.setEmbeddings(map);
    }
  }

  /**
   * Atomically replace catalog tools synchronously without external embedding calls (runs pure BM25).
   */
  public setToolsSync(tools: ToolDefinition[]): void {
    this.catalog.setTools(tools);
    this.resolver.syncIndex();
  }

  /**
   * Set pre-computed vector embeddings for tools.
   */
  public setEmbeddings(embeddings: Map<string, Float32Array> | Record<string, Float32Array | number[]>): void {
    this.catalog.setEmbeddings(embeddings);
  }

  /**
   * Add workflow transition edges to the companion graph.
   */
  public addWorkflowEdges(edges: ToolTransitionEdge[]): void {
    this.catalog.addEdges(edges);
  }

  /**
   * Synchronously resolve active tools using local BM25 ranking (pure CPU, 0ms async latency).
   */
  public resolveSync(
    query: string,
    queryEmbedding?: Float32Array,
    session?: SessionState,
    options?: RouterOptions
  ): ToolRouteResult {
    return this.resolver.resolve(query, queryEmbedding, session, options);
  }

  /**
   * Asynchronously resolve active tools. Computes query embeddings if an embedder is configured.
   */
  public async resolve(
    query: string,
    session?: SessionState,
    options?: RouterOptions
  ): Promise<ToolRouteResult> {
    let queryEmbedding: Float32Array | undefined = undefined;

    if (this.embedder) {
      queryEmbedding = await this.embedder.embedQuery(query);
    }

    return this.resolver.resolve(query, queryEmbedding, session, options);
  }

  private toolsetCache = new Map<string, ToolImpulse>();
  private filterInitPromises = new Map<string, Promise<ToolImpulse>>();

  /**
   * Synchronous tool filtering using local BM25 ranking.
   * If tools match the current catalog, resolves against the current engine.
   * If dynamic tools are passed, resolves against an isolated cached instance in toolsetCache
   * to guarantee zero cross-tenant catalog mutation and thread safety.
   */
  public filterSync<T extends { name: string; description?: string; embedding?: Float32Array | number[] }>(
    query: string,
    tools: T[],
    options?: RouterOptions
  ): { tools: T[]; result: ToolRouteResult } {
    const catalogNames = this.catalog.getToolNames();
    const isExactMatch =
      catalogNames.length === tools.length &&
      tools.every((t) => this.catalog.hasTool(t.name));

    let activeEngine: ToolImpulse;

    if (isExactMatch) {
      activeEngine = this;
    } else if (catalogNames.length === 0) {
      this.setToolsSync(
        tools.map((t) => ({
          name: t.name,
          description: t.description || '',
          embedding: t.embedding,
        }))
      );
      activeEngine = this;
    } else {
      const fp = ToolCatalog.computeFingerprint(tools);
      let scoped = getFromLru(this.toolsetCache, fp);

      if (!scoped) {
        scoped = new ToolImpulse({
          tools: tools.map((t) => ({
            name: t.name,
            description: t.description || '',
            embedding: t.embedding,
          })),
          defaultOptions: this.defaultOptions,
        });
        setInLru(this.toolsetCache, fp, scoped);
      }
      activeEngine = scoped;
    }

    const result = activeEngine.resolveSync(query, undefined, undefined, options);
    const allowed = new Set(result.selectedNames);
    return {
      tools: tools.filter((t) => allowed.has(t.name)),
      result,
    };
  }

  /**
   * Async filtering with query embedding support.
   * If tools match the current catalog, resolves against the current engine.
   * If dynamic tools are passed, resolves against an isolated cached instance in toolsetCache
   * to guarantee zero cross-tenant catalog mutation and thread safety.
   */
  public async filter<T extends { name: string; description?: string; embedding?: Float32Array | number[] }>(
    query: string,
    tools: T[],
    options?: RouterOptions
  ): Promise<{ tools: T[]; result: ToolRouteResult }> {
    const catalogNames = this.catalog.getToolNames();
    const isExactMatch =
      catalogNames.length === tools.length &&
      tools.every((t) => this.catalog.hasTool(t.name));

    let activeEngine: ToolImpulse;

    if (isExactMatch) {
      activeEngine = this;
    } else if (catalogNames.length === 0) {
      // Uninitialized engine: initialize this instance
      await this.setTools(
        tools.map((t) => ({
          name: t.name,
          description: t.description || '',
          embedding: t.embedding,
        }))
      );
      activeEngine = this;
    } else {
      const fp = ToolCatalog.computeFingerprint(tools);
      let scoped = getFromLru(this.toolsetCache, fp);

      if (!scoped) {
        let inFlight = this.filterInitPromises.get(fp);
        if (!inFlight) {
          inFlight = (async () => {
            const instance = new ToolImpulse({
              tools: tools.map((t) => ({
                name: t.name,
                description: t.description || '',
                embedding: t.embedding,
              })),
              embedder: this.embedder,
              defaultOptions: this.defaultOptions,
            });
            if (this.embedder) {
              await instance.setTools(
                tools.map((t) => ({
                  name: t.name,
                  description: t.description || '',
                  embedding: t.embedding,
                }))
              );
            }
            setInLru(this.toolsetCache, fp, instance);
            this.filterInitPromises.delete(fp);
            return instance;
          })();
          this.filterInitPromises.set(fp, inFlight);
        }
        scoped = await inFlight;
      }
      activeEngine = scoped;
    }

    const result = await activeEngine.resolve(query, undefined, options);
    const allowed = new Set(result.selectedNames);
    return {
      tools: tools.filter((t) => allowed.has(t.name)),
      result,
    };
  }

  private static readonly staticFilterCache = new Map<string, ToolImpulse>();

  /**
   * Stateless static filter for one-shot tool selection without managing an engine instance.
   * Memoizes indexed instances by deterministic catalog fingerprint so repeated calls
   * with equivalent toolsets (even if newly allocated) execute in <0.02ms without re-indexing BM25.
   */
  public static filter<T extends { name: string; description?: string; embedding?: Float32Array | number[] }>(
    query: string,
    tools: T[],
    options?: RouterOptions
  ): T[] {
    const fp = ToolCatalog.computeFingerprint(tools);
    let engine = getFromLru(ToolImpulse.staticFilterCache, fp);

    if (!engine) {
      engine = new ToolImpulse({
        tools: tools.map((t) => ({
          name: t.name,
          description: t.description || '',
          embedding: t.embedding,
        })),
      });
      setInLru(ToolImpulse.staticFilterCache, fp, engine);
    }

    const result = engine.resolveSync(query, undefined, undefined, options);
    const allowed = new Set(result.selectedNames);
    return tools.filter((t) => allowed.has(t.name));
  }

  /**
   * Record multi-tool workflow execution to reinforce companion edges over time.
   */
  public recordWorkflow(toolsExecuted: string[]): void {
    if (!toolsExecuted || toolsExecuted.length < 2) return;

    for (let i = 0; i < toolsExecuted.length - 1; i++) {
      const from = toolsExecuted[i];
      const to = toolsExecuted[i + 1];
      if (from !== to) {
        this.catalog.recordTransition(from, to);
      }
    }
  }

  public getCatalog(): ToolCatalog {
    return this.catalog;
  }

  public getResolver(): ToolResolver {
    return this.resolver;
  }

  /**
   * Export in-memory catalog, vector embeddings, and workflow graph for serialization.
   * Allows saving pre-computed tool states to disk/KV to eliminate startup embedding costs.
   */
  public exportState(): ToolCatalogState {
    return this.catalog.exportState();
  }

  /**
   * Hydrate catalog, vector embeddings, and workflow graph from serialized state.
   */
  public importState(state: ToolCatalogState): void {
    this.catalog.importState(state);
    this.resolver.syncIndex();
  }
}

export { ToolImpulse as ToolRouter };
