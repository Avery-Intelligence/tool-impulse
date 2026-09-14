import { ToolCatalogState, ToolDefinition, ToolTransitionEdge } from './types.js';

export interface ToolCatalogEntry {
  tool: ToolDefinition;
  embedding?: Float32Array;
}

export class ToolCatalog {
  private entries: Map<string, ToolCatalogEntry> = new Map();
  private toolNames: string[] = [];
  private graph: Map<string, Map<string, number>> = new Map();
  private transitionCounts: Map<string, Map<string, number>> = new Map();
  private dimension: number = 0;

  /**
   * Register tools into the in-memory catalog.
   * If a tool includes an embedding, it is normalized and stored.
   */
  public registerTools(tools: ToolDefinition[]): void {
    for (const tool of tools) {
      if (!this.entries.has(tool.name)) {
        this.toolNames.push(tool.name);
      }

      if (!tool.domain) {
        tool.domain = tool.name.includes('_') ? tool.name.split('_')[0] : 'default';
      }

      const entry: ToolCatalogEntry = { tool };
      if (tool.embedding) {
        entry.embedding = ToolCatalog.normalizeVector(tool.embedding);
        if (this.dimension === 0 && entry.embedding.length > 0) {
          this.dimension = entry.embedding.length;
        }
      }

      this.entries.set(tool.name, entry);
    }
  }

  /**
   * Normalize an embedding vector to unit length (L2 norm).
   */
  public static normalizeVector(raw: Float32Array | number[]): Float32Array {
    const vec = raw instanceof Float32Array ? raw : new Float32Array(raw);
    let sumSq = 0;
    for (let i = 0; i < vec.length; i++) {
      sumSq += vec[i] * vec[i];
    }
    const norm = Math.sqrt(sumSq) || 1e-10;
    const normalized = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
      normalized[i] = vec[i] / norm;
    }
    return normalized;
  }

  /**
   * Clear all registered tools, embeddings, and workflow edges.
   */
  public clear(): void {
    this.entries.clear();
    this.toolNames = [];
    this.graph.clear();
    this.transitionCounts.clear();
    this.dimension = 0;
  }

  /**
   * Store pre-computed or generated vector embeddings (unit normalized).
   */
  public setEmbeddings(embeddings: Map<string, Float32Array> | Record<string, Float32Array | number[]>): void {
    const items = embeddings instanceof Map ? embeddings.entries() : Object.entries(embeddings);

    for (const [name, raw] of items) {
      const entry = this.entries.get(name);
      if (!entry) continue;

      const vec = raw instanceof Float32Array ? raw : new Float32Array(raw);
      if (this.dimension === 0 && vec.length > 0) {
        this.dimension = vec.length;
      }

      // Unit normalize L2 length
      let sumSq = 0;
      for (let i = 0; i < vec.length; i++) {
        sumSq += vec[i] * vec[i];
      }
      const norm = Math.sqrt(sumSq) || 1e-10;
      const normalized = new Float32Array(vec.length);
      for (let i = 0; i < vec.length; i++) {
        normalized[i] = vec[i] / norm;
      }

      entry.embedding = normalized;
    }
  }

  /**
   * Add directed transition edges to the companion tool workflow graph.
   */
  public addEdges(edges: ToolTransitionEdge[]): void {
    for (const edge of edges) {
      this.setEdge(edge.fromTool, edge.toTool, edge.weight);
    }
  }

  public setEdge(fromTool: string, toTool: string, weight: number): void {
    let neighbors = this.graph.get(fromTool);
    if (!neighbors) {
      neighbors = new Map();
      this.graph.set(fromTool, neighbors);
    }
    neighbors.set(toTool, Math.max(0, Math.min(1.0, weight)));
  }

  /**
   * Update workflow transition graph based on actual multi-step execution receipts.
   */
  public recordTransition(fromTool: string, toTool: string, priorWeight: number = 2.0): void {
    let counts = this.transitionCounts.get(fromTool);
    if (!counts) {
      counts = new Map();
      this.transitionCounts.set(fromTool, counts);
    }
    const count = (counts.get(toTool) || 0) + 1;
    counts.set(toTool, count);

    let total = 0;
    for (const n of counts.values()) {
      total += n;
    }

    const currentEdge = this.getEdgeWeight(fromTool, toTool) || 0.1;
    const updatedWeight = (priorWeight * currentEdge + count) / (priorWeight + total);
    this.setEdge(fromTool, toTool, updatedWeight);
  }

  public getEdgeWeight(fromTool: string, toTool: string): number {
    return this.graph.get(fromTool)?.get(toTool) || 0.0;
  }

  public getNeighbors(toolName: string): Map<string, number> {
    return this.graph.get(toolName) || new Map();
  }

  public getTool(name: string): ToolDefinition | undefined {
    return this.entries.get(name)?.tool;
  }

  public getAllTools(): ToolDefinition[] {
    return Array.from(this.entries.values()).map((e) => e.tool);
  }

  public getToolNames(): string[] {
    return this.toolNames;
  }

  public getDimension(): number {
    return this.dimension;
  }

  /**
   * Compute cosine similarity between normalized query vector and tool vector.
   */
  public computeCosine(queryVec: Float32Array, toolName: string): number {
    const entry = this.entries.get(toolName);
    if (!entry || !entry.embedding) return 0.0;

    const vec = entry.embedding;
    let dot = 0.0;
    const len = Math.min(queryVec.length, vec.length);

    for (let i = 0; i < len; i++) {
      dot += queryVec[i] * vec[i];
    }

    return Math.max(0.0, Math.min(1.0, dot));
  }

  /**
   * Export in-memory catalog, vector embeddings, and workflow graph state for serialization.
   */
  public exportState(): ToolCatalogState {
    const tools = this.getAllTools();
    const embeddings: Record<string, number[]> = {};

    for (const [name, entry] of this.entries) {
      if (entry.embedding) {
        embeddings[name] = Array.from(entry.embedding);
      }
    }

    const edges: ToolTransitionEdge[] = [];
    for (const [from, toMap] of this.graph) {
      for (const [to, weight] of toMap) {
        edges.push({ fromTool: from, toTool: to, weight });
      }
    }

    return {
      version: 1,
      tools,
      embeddings: Object.keys(embeddings).length > 0 ? embeddings : undefined,
      edges: edges.length > 0 ? edges : undefined,
    };
  }

  /**
   * Hydrate catalog, vector embeddings, and workflow graph from serialized state.
   */
  public importState(state: ToolCatalogState): void {
    if (!state || !state.tools) return;

    this.registerTools(state.tools);

    if (state.embeddings) {
      this.setEmbeddings(state.embeddings);
    }

    if (state.edges) {
      this.addEdges(state.edges);
    }
  }
}
