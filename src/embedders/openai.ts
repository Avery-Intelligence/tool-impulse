import { EmbeddingProvider } from '../core/types.js';

export interface OpenAIEmbedderConfig {
  apiKey: string;
  model?: string;
  dimensions?: number;
  baseUrl?: string;
}

export class OpenAIEmbedder implements EmbeddingProvider {
  private apiKey: string;
  private model: string;
  public readonly dimension: number;
  private baseUrl: string;

  constructor(config: OpenAIEmbedderConfig) {
    this.apiKey = config.apiKey;
    this.model = config.model || 'text-embedding-3-small';
    this.dimension = config.dimensions || 1536;
    this.baseUrl = config.baseUrl || 'https://api.openai.com/v1';
  }

  public async embedQuery(text: string): Promise<Float32Array> {
    const vectors = await this.callApi([text]);
    return vectors[0];
  }

  public async embedBatch(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];

    const chunkSize = 50;
    const allVectors: Float32Array[] = [];

    for (let i = 0; i < texts.length; i += chunkSize) {
      const chunk = texts.slice(i, i + chunkSize);
      const batchVectors = await this.callApi(chunk);
      allVectors.push(...batchVectors);
    }

    return allVectors;
  }

  private async callApi(inputs: string[]): Promise<Float32Array[]> {
    const response = await fetch(`${this.baseUrl}/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        input: inputs,
        model: this.model,
        dimensions: this.dimension,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI Embedding API error (${response.status}): ${errorText}`);
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[]; index: number }>;
    };

    data.data.sort((a, b) => a.index - b.index);
    return data.data.map((item) => new Float32Array(item.embedding));
  }
}
