import { ImpulseEmbedder } from '../core/types.js';

/**
 * Deterministic local hashing embedder.
 * Projects lexical tokens into a pseudo-dense Float32Array using feature hashing (hash trick).
 * Requires zero network calls, zero API keys, and runs in microseconds.
 */
export class LocalHashingEmbedder implements ImpulseEmbedder {
  public readonly dimension: number;

  constructor(dimension: number = 384) {
    this.dimension = dimension;
  }

  public async embedQuery(text: string): Promise<Float32Array> {
    return this.hashText(text);
  }

  public async embedBatch(texts: string[]): Promise<Float32Array[]> {
    return texts.map((t) => this.hashText(t));
  }

  private hashText(text: string): Float32Array {
    const vec = new Float32Array(this.dimension);
    const tokens = text
      .toLowerCase()
      .split(/[^a-z0-9_]+/)
      .filter((t) => t.length > 1);

    if (tokens.length === 0) return vec;

    let sumSq = 0;
    for (const token of tokens) {
      // 32-bit FNV-1a hash
      let h = 2166136261;
      for (let i = 0; i < token.length; i++) {
        h ^= token.charCodeAt(i);
        h = Math.imul(h, 16777619);
      }

      const index = Math.abs(h) % this.dimension;
      const sign = (h & 0x1000) === 0 ? 1.0 : -1.0;
      vec[index] += sign;
    }

    // Normalize
    for (let i = 0; i < this.dimension; i++) {
      sumSq += vec[i] * vec[i];
    }
    const norm = Math.sqrt(sumSq) || 1e-10;
    for (let i = 0; i < this.dimension; i++) {
      vec[i] /= norm;
    }

    return vec;
  }
}
