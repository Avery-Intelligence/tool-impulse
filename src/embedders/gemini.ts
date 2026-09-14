import { EmbeddingProvider } from '../core/types.js';

export interface GeminiEmbedderConfig {
  apiKey: string;
  model?: string;
  dimensions?: number;
  baseUrl?: string;
}

/**
 * Native Google Gemini embedding provider (using `text-embedding-004`).
 * Zero external dependencies, pure native `fetch`.
 *
 * @example
 * ```typescript
 * import { ToolImpulse, GeminiEmbedder } from 'tool-impulse';
 *
 * const embedder = new GeminiEmbedder({
 *   apiKey: process.env.GEMINI_API_KEY!,
 * });
 *
 * const engine = new ToolImpulse({ embedder });
 * await engine.registerTools(myTools);
 * ```
 */
export class GeminiEmbedder implements EmbeddingProvider {
  private apiKey: string;
  private model: string;
  public readonly dimension: number;
  private baseUrl: string;

  constructor(config: GeminiEmbedderConfig) {
    this.apiKey = config.apiKey;
    this.model = config.model || 'text-embedding-004';
    this.dimension = config.dimensions || 768;
    this.baseUrl = config.baseUrl || 'https://generativelanguage.googleapis.com/v1beta';
  }

  public async embedQuery(text: string): Promise<Float32Array> {
    const vectors = await this.embedBatch([text]);
    return vectors[0];
  }

  public async embedBatch(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];

    const chunkSize = 20; // Gemini recommended batch limit
    const allVectors: Float32Array[] = [];

    for (let i = 0; i < texts.length; i += chunkSize) {
      const chunk = texts.slice(i, i + chunkSize);
      const batchVectors = await this.callBatchApi(chunk);
      allVectors.push(...batchVectors);
    }

    return allVectors;
  }

  private async callBatchApi(inputs: string[]): Promise<Float32Array[]> {
    const modelPath = this.model.startsWith('models/') ? this.model : `models/${this.model}`;
    const url = `${this.baseUrl}/${modelPath}:batchEmbedContents?key=${encodeURIComponent(this.apiKey)}`;

    const requests = inputs.map((text) => ({
      model: modelPath,
      content: {
        parts: [{ text }],
      },
      outputDimensionality: this.dimension,
    }));

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ requests }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Gemini Embedding API error (${response.status}): ${errorText}`);
    }

    const data = (await response.json()) as {
      embeddings?: Array<{ values: number[] }>;
    };

    if (!data.embeddings || !Array.isArray(data.embeddings)) {
      throw new Error('Malformed response from Gemini Embedding API: missing embeddings array');
    }

    return data.embeddings.map((item) => new Float32Array(item.values));
  }
}
