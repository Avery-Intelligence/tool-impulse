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
   * Atomically replace catalog tools and synchronize the Okapi BM25 lexical index.
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
   * Define workflow companion edges between tools.
   */
  public addWorkflowEdges(edges: ToolTransitionEdge[]): void {
    this.catalog.addEdges(edges);
  }

  /**
   * Resolve active tools for the user query.
   * Thread-safe and read-only: safe to call concurrently from multiple requests.
   */
  public async resolve(
    query: string,
    session?: SessionState,
    options?: RouterOptions
  ): Promise<ToolRouteResult> {
    const opts = { ...this.defaultOptions, ...options };
    let queryEmbedding: Float32Array | undefined = undefined;

    if (this.embedder) {
      queryEmbedding = await this.embedder.embedQuery(query);
    }

    return this.resolver.resolve(query, queryEmbedding, session, opts);
  }

  /**
   * Synchronous resolution (when running in pure BM25 mode or with pre-computed query vectors).
   * Thread-safe and read-only.
   */
  public resolveSync(
    query: string,
    queryEmbedding?: Float32Array,
    session?: SessionState,
    options?: RouterOptions
  ): ToolRouteResult {
    const opts = { ...this.defaultOptions, ...options };
    return this.resolver.resolve(query, queryEmbedding, session, opts);
  }

  /**
   * Filter an array of tools down to the relevant subset for the query.
   * Preserves the exact input tool type T. Re-synchronizes catalog if tools change dynamically.
   */
  public filterSync<T extends { name: string; description?: string; embedding?: Float32Array | number[] }>(
    query: string,
    tools: T[],
    options?: RouterOptions
  ): { tools: T[]; result: ToolRouteResult } {
    const catalogNames = this.catalog.getToolNames();
    const needsSync =
      catalogNames.length !== tools.length ||
      tools.some((t) => !this.catalog.hasTool(t.name));

    if (needsSync) {
      this.setToolsSync(
        tools.map((t) => ({
          name: t.name,
          description: t.description || '',
          embedding: t.embedding,
        }))
      );
    }
    const result = this.resolveSync(query, undefined, undefined, options);
    const allowed = new Set(result.selectedNames);
    return {
      tools: tools.filter((t) => allowed.has(t.name)),
      result,
    };
  }

  /**
   * Async filtering with query embedding support.
   * Re-synchronizes catalog if tools change dynamically.
   */
  public async filter<T extends { name: string; description?: string; embedding?: Float32Array | number[] }>(
    query: string,
    tools: T[],
    options?: RouterOptions
  ): Promise<{ tools: T[]; result: ToolRouteResult }> {
    const catalogNames = this.catalog.getToolNames();
    const needsSync =
      catalogNames.length !== tools.length ||
      tools.some((t) => !this.catalog.hasTool(t.name));

    if (needsSync) {
      this.catalog.setTools(
        tools.map((t) => ({
          name: t.name,
          description: t.description || '',
          embedding: t.embedding,
        }))
      );
      this.resolver.syncIndex();

      if (this.embedder) {
        const descriptions = tools.map((t) => `${t.name}: ${t.description || ''}`);
        const vectors = await this.embedder.embedBatch(descriptions);
        const map = new Map<string, Float32Array>();
        for (let i = 0; i < tools.length; i++) {
          map.set(tools[i].name, vectors[i]);
        }
        this.catalog.setEmbeddings(map);
      }
    }
    const result = await this.resolve(query, undefined, options);
    const allowed = new Set(result.selectedNames);
    return {
      tools: tools.filter((t) => allowed.has(t.name)),
      result,
    };
  }

  /**
   * Stateless static filter for one-shot tool selection without managing an engine instance.
   * Creates an isolated, ephemeral scoped catalog: 100% concurrency-safe.
   */
  public static filter<T extends { name: string; description?: string; embedding?: Float32Array | number[] }>(
    query: string,
    tools: T[],
    options?: RouterOptions
  ): T[] {
    const engine = new ToolImpulse({
      tools: tools.map((t) => ({
        name: t.name,
        description: t.description || '',
        embedding: t.embedding,
      })),
    });
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
