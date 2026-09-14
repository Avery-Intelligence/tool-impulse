import { ImpulseBank } from './bank.js';
import { ImpulseResolver } from './resolver.js';
import {
  ImpulseEmbedder,
  ImpulseOptions,
  ImpulseResult,
  ImpulseSessionState,
  ImpulseTool,
  ToolTransitionEdge,
} from './types.js';

export interface ToolImpulseEngineConfig {
  embedder?: ImpulseEmbedder;
  defaultOptions?: ImpulseOptions;
  initialTools?: ImpulseTool[];
  initialEdges?: ToolTransitionEdge[];
}

export class ToolImpulseEngine {
  private bank: ImpulseBank;
  private resolver: ImpulseResolver;
  private embedder?: ImpulseEmbedder;
  private defaultOptions?: ImpulseOptions;

  constructor(config: ToolImpulseEngineConfig = {}) {
    this.bank = new ImpulseBank();
    this.resolver = new ImpulseResolver(this.bank);
    this.embedder = config.embedder;
    this.defaultOptions = config.defaultOptions;

    if (config.initialTools) {
      this.registerToolsSync(config.initialTools);
    }
    if (config.initialEdges) {
      this.bank.addEdges(config.initialEdges);
    }
  }

  /**
   * Register tools and compute embeddings (if an embedder is configured).
   * Also indexes tools into the Okapi BM25 sparse index.
   */
  public async registerTools(tools: ImpulseTool[]): Promise<void> {
    this.bank.registerTools(tools);
    this.resolver.syncBm25Index();

    if (this.embedder) {
      const descriptions = tools.map((t) => `${t.name}: ${t.description} ${(t.keywords || []).join(' ')}`);
      const vectors = await this.embedder.embedBatch(descriptions);
      const map = new Map<string, Float32Array>();
      for (let i = 0; i < tools.length; i++) {
        map.set(tools[i].name, vectors[i]);
      }
      this.bank.setEmbeddings(map);
    }
  }

  /**
   * Synchronous tool registration without computing embeddings (runs in fast Okapi BM25 mode).
   */
  public registerToolsSync(tools: ImpulseTool[]): void {
    this.bank.registerTools(tools);
    this.resolver.syncBm25Index();
  }

  /**
   * Set pre-computed embeddings for tools.
   */
  public setEmbeddings(embeddings: Map<string, Float32Array> | Record<string, Float32Array | number[]>): void {
    this.bank.setEmbeddings(embeddings);
  }

  /**
   * Add directed transition edges to the companion graph.
   */
  public addEdges(edges: ToolTransitionEdge[]): void {
    this.bank.addEdges(edges);
  }

  /**
   * Resolve the active tools for an incoming user query.
   */
  public async resolve(
    query: string,
    session?: ImpulseSessionState,
    options?: ImpulseOptions
  ): Promise<ImpulseResult> {
    const opts = { ...this.defaultOptions, ...options };
    let queryEmbedding: Float32Array | undefined = undefined;

    if (this.embedder) {
      queryEmbedding = await this.embedder.embedQuery(query);
    }

    return this.resolver.resolve(query, queryEmbedding, session, opts);
  }

  /**
   * Synchronous resolution (when pre-computed embeddings exist or using BM25 lexical mode).
   */
  public resolveSync(
    query: string,
    queryEmbedding?: Float32Array,
    session?: ImpulseSessionState,
    options?: ImpulseOptions
  ): ImpulseResult {
    const opts = { ...this.defaultOptions, ...options };
    return this.resolver.resolve(query, queryEmbedding, session, opts);
  }

  /**
   * Record multi-tool execution telemetry to reinforce companion edges.
   */
  public recordExecution(toolsUsed: string[]): void {
    if (!toolsUsed || toolsUsed.length < 2) return;

    for (let i = 0; i < toolsUsed.length - 1; i++) {
      const from = toolsUsed[i];
      const to = toolsUsed[i + 1];
      if (from !== to) {
        this.bank.recordTransition(from, to);
      }
    }
  }

  public getBank(): ImpulseBank {
    return this.bank;
  }

  public getResolver(): ImpulseResolver {
    return this.resolver;
  }
}
