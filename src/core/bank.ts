import { ImpulseTool, ToolTransitionEdge } from './types.js';
import { Int8Quantizer, QuantizedVector } from './quantization.js';

export interface BankToolEntry {
  tool: ImpulseTool;
  embedding?: Float32Array;
  quantized?: QuantizedVector;
  norm: number;
}

export interface ImpulseBankConfig {
  /** Enable Int8 scalar quantization to reduce vector memory by 75% */
  quantizeInt8?: boolean;
}

export class ImpulseBank {
  private entries: Map<string, BankToolEntry> = new Map();
  private toolNames: string[] = [];
  private graph: Map<string, Map<string, number>> = new Map();
  private observationCounts: Map<string, Map<string, number>> = new Map();
  private dimension: number = 0;
  private quantizeInt8: boolean;

  constructor(config: ImpulseBankConfig = {}) {
    this.quantizeInt8 = !!config.quantizeInt8;
  }

  /**
   * Register tools into the in-memory bank.
   */
  public registerTools(tools: ImpulseTool[]): void {
    for (const tool of tools) {
      if (!this.entries.has(tool.name)) {
        this.toolNames.push(tool.name);
      }
      this.entries.set(tool.name, {
        tool,
        norm: 1.0,
      });

      // Auto-extract family if not provided (e.g. "stripe_charge" -> "stripe")
      if (!tool.family) {
        const parts = tool.name.split('_');
        tool.family = parts.length > 1 ? parts[0] : 'default';
      }
    }
  }

  /**
   * Set pre-computed or newly generated embeddings for tools.
   * Vectors are automatically normalized to unit L2 length and optionally quantized.
   */
  public setEmbeddings(embeddings: Map<string, Float32Array> | Record<string, Float32Array | number[]>): void {
    const entries = embeddings instanceof Map ? embeddings.entries() : Object.entries(embeddings);

    for (const [name, rawVec] of entries) {
      const entry = this.entries.get(name);
      if (!entry) continue;

      const vec = rawVec instanceof Float32Array ? rawVec : new Float32Array(rawVec);
      if (this.dimension === 0 && vec.length > 0) {
        this.dimension = vec.length;
      }

      // Compute L2 norm and normalize
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
      entry.norm = 1.0;

      if (this.quantizeInt8) {
        entry.quantized = Int8Quantizer.quantize(normalized);
      }
    }
  }

  /**
   * Add directed transition edges to the companion graph.
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
   * Online Bayesian update of edge weight based on multi-tool execution telemetry.
   * Uses Dirichlet-Multinomial conjugate updating:
   * W(i, j) = (alpha_0 * W_0 + N(i, j)) / (alpha_0 + sum_k N(i, k))
   */
  public recordTransition(fromTool: string, toTool: string, priorWeight: number = 2.0): void {
    let counts = this.observationCounts.get(fromTool);
    if (!counts) {
      counts = new Map();
      this.observationCounts.set(fromTool, counts);
    }
    const current = counts.get(toTool) || 0;
    counts.set(toTool, current + 1);

    // Compute total observations from `fromTool`
    let totalObs = 0;
    for (const n of counts.values()) {
      totalObs += n;
    }

    const currentEdgeWeight = this.getEdgeWeight(fromTool, toTool) || 0.1;
    const updatedWeight = (priorWeight * currentEdgeWeight + (current + 1)) / (priorWeight + totalObs);
    this.setEdge(fromTool, toTool, updatedWeight);
  }

  public getEdgeWeight(fromTool: string, toTool: string): number {
    return this.graph.get(fromTool)?.get(toTool) || 0.0;
  }

  public getNeighbors(toolName: string): Map<string, number> {
    return this.graph.get(toolName) || new Map();
  }

  public getTool(name: string): ImpulseTool | undefined {
    return this.entries.get(name)?.tool;
  }

  public getAllTools(): ImpulseTool[] {
    return Array.from(this.entries.values()).map((e) => e.tool);
  }

  public getAllEntries(): Map<string, BankToolEntry> {
    return this.entries;
  }

  public getToolNames(): string[] {
    return this.toolNames;
  }

  public getDimension(): number {
    return this.dimension;
  }

  public isQuantized(): boolean {
    return this.quantizeInt8;
  }

  /**
   * Computes cosine similarity between a normalized query vector and a tool vector.
   * Uses Int8 integer math if bank is quantized; otherwise unrolled Float32 dot product.
   */
  public computeCosine(queryVec: Float32Array, toolName: string): number {
    const entry = this.entries.get(toolName);
    if (!entry) return 0.0;

    if (this.quantizeInt8 && entry.quantized) {
      return Int8Quantizer.dotProductWithFloat(queryVec, entry.quantized);
    }

    if (!entry.embedding) return 0.0;

    const vec = entry.embedding;
    const len = vec.length;
    let dot = 0.0;

    // Loop with 4x unrolling for performance
    let i = 0;
    const limit = len - 3;
    for (; i < limit; i += 4) {
      dot += queryVec[i] * vec[i] +
             queryVec[i + 1] * vec[i + 1] +
             queryVec[i + 2] * vec[i + 2] +
             queryVec[i + 3] * vec[i + 3];
    }
    for (; i < len; i++) {
      dot += queryVec[i] * vec[i];
    }

    return Math.max(0.0, Math.min(1.0, dot));
  }
}
