import { EmbeddingProvider } from '../core/types.js';

export interface CloudflareAiBinding {
  run: (model: string, input: { text: string | string[] }) => Promise<any>;
}

export interface CloudflareEmbedderConfig {
  /** Cloudflare `env.AI` binding when running inside a Cloudflare Worker */
  ai?: CloudflareAiBinding;
  /** Cloudflare Account ID for REST API calls */
  accountId?: string;
  /** Cloudflare API Token for REST API calls */
  apiToken?: string;
  /** Embedding model (default: '@cf/baai/bge-small-en-v1.5') */
  model?: string;
  /** Output vector dimension (default: 384 for bge-small, 768 for bge-base) */
  dimensions?: number;
}

/**
 * Cloudflare Workers AI embedding provider.
 * Supports both direct `env.AI` binding inside Workers and REST API access from Node.js.
 *
 * @example Inside Cloudflare Worker:
 * ```typescript
 * import { ToolImpulse, CloudflareEmbedder } from 'tool-impulse';
 *
 * export default {
 *   async fetch(request, env) {
 *     const embedder = new CloudflareEmbedder({ ai: env.AI });
 *     const engine = new ToolImpulse({ embedder });
 *     // ...
 *   }
 * }
 * ```
 */
export class CloudflareEmbedder implements EmbeddingProvider {
  private ai?: CloudflareAiBinding;
  private accountId?: string;
  private apiToken?: string;
  private model: string;
  public readonly dimension: number;

  constructor(config: CloudflareEmbedderConfig = {}) {
    this.ai = config.ai;
    this.accountId = config.accountId;
    this.apiToken = config.apiToken;
    this.model = config.model || '@cf/baai/bge-small-en-v1.5';
    this.dimension = config.dimensions || 384;

    if (!this.ai && (!this.accountId || !this.apiToken)) {
      // Allowed to construct; embedBatch will throw clear error if invoked without credentials
    }
  }

  public async embedQuery(text: string): Promise<Float32Array> {
    const vectors = await this.embedBatch([text]);
    return vectors[0];
  }

  public async embedBatch(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];

    if (this.ai) {
      return this.callBinding(texts);
    }

    if (this.accountId && this.apiToken) {
      return this.callRestApi(texts);
    }

    throw new Error(
      'CloudflareEmbedder requires either an `env.AI` binding or both `accountId` and `apiToken`.'
    );
  }

  private async callBinding(texts: string[]): Promise<Float32Array[]> {
    const response = await this.ai!.run(this.model, { text: texts });
    const rawData = response.data || response;

    if (Array.isArray(rawData)) {
      return rawData.map((vec: number[]) => new Float32Array(vec));
    }

    throw new Error('Unexpected response format from Cloudflare env.AI binding');
  }

  private async callRestApi(texts: string[]): Promise<Float32Array[]> {
    const url = `https://api.cloudflare.com/client/v4/accounts/${this.accountId}/ai/run/${this.model}`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: texts }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Cloudflare AI REST API error (${response.status}): ${errorText}`);
    }

    const data = (await response.json()) as {
      result?: { data?: number[][] };
      success?: boolean;
      errors?: Array<{ message: string }>;
    };

    if (!data.success || !data.result?.data) {
      const msg = data.errors?.[0]?.message || 'Unknown Cloudflare AI error';
      throw new Error(`Cloudflare AI execution failed: ${msg}`);
    }

    return data.result.data.map((vec) => new Float32Array(vec));
  }
}
