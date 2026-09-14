/**
 * Core type definitions for Tool Impulse.
 */

export interface ToolDefinition {
  /** Unique tool identifier (e.g. "stripe_list_invoices", "jira_update_issue") */
  name: string;
  /** Natural language description explaining when and how to use the tool */
  description: string;
  /** Optional JSON Schema parameter specification */
  parameters?: Record<string, unknown>;
  /** Domain / service grouping (e.g. "stripe", "jira", "slack"). Inferred from prefix if omitted. */
  domain?: string;
  /** Optional keywords for exact lexical matching */
  keywords?: string[];
  /** Mark read-only or idempotent operations */
  readOnly?: boolean;
  /** Optional pre-computed vector embedding (Float32Array or number array) */
  embedding?: Float32Array | number[];
}

export interface SessionState {
  /** Embedding of the prior user turn for trajectory context blending */
  priorTurnEmbedding?: Float32Array | number[];
  /** Tool names executed in recent turns (receives inertia bonus) */
  recentToolNames?: string[];
  /** Turn index in conversation */
  turnCount?: number;
}

export interface ToolTransitionEdge {
  fromTool: string;
  toTool: string;
  weight: number;
}

export interface ToolCatalogState {
  version: 1;
  tools: ToolDefinition[];
  embeddings?: Record<string, number[]>;
  edges?: ToolTransitionEdge[];
}

export interface EmbeddingProvider {
  /** Vector dimension (e.g. 768 for Gemini, 1536 for OpenAI) */
  readonly dimension: number;
  /** Generate embedding for a single text query */
  embedQuery(text: string): Promise<Float32Array>;
  /** Generate embeddings for a batch of strings */
  embedBatch(texts: string[]): Promise<Float32Array[]>;
}

export interface RouterOptions {
  /** Maximum number of active tools to return for this turn (default: 3) */
  topK?: number;
  /** Maximum tools allowed from a single domain (uncapped by default) */
  maxPerDomain?: number;
  /** Weight for vector similarity vs BM25: alpha * dense + (1 - alpha) * BM25 (default: 0.70) */
  alpha?: number;
  /** Multi-turn trajectory blend factor: beta * current + (1 - beta) * prior (default: 0.75) */
  beta?: number;
  /** Minimum cosine similarity between current and prior turn to blend trajectory. If below threshold, topic shift is detected and prior trajectory is discarded (default: 0.35) */
  driftThreshold?: number;
  /** Custom domain resolver function to group tools for maxPerDomain capping */
  domainResolver?: (tool: ToolDefinition) => string | undefined;
  /** Score boost for tools executed in recent turns (default: 0.20) */
  inertiaBonus?: number;
  /** Score multiplier for companion tools connected in the workflow graph (default: 0.25) */
  companionBoost?: number;
  /** Minimum score required to mount a tool (default: 0.05) */
  minScoreThreshold?: number;
  /** Fallback tools to return when the user query matches nothing (e.g. greetings) */
  defaultTools?: string[];
  /** Enable human-readable explanation trace for debugging */
  debug?: boolean;
}

export interface ScoredToolMatch {
  tool: ToolDefinition;
  totalScore: number;
  denseScore: number;
  bm25Score: number;
  companionBonus: number;
  inertiaBonus: number;
  domain: string;
  reasons: string[];
}

export interface ToolRouteResult {
  /** Active tools selected for the prompt (length <= topK) */
  tools: ToolDefinition[];
  /** Exact names of selected tools */
  selectedNames: string[];
  /** Primary top-1 matched tool */
  primaryTool?: ToolDefinition;
  /** Contextualized query embedding for this turn (pass to next turn's session) */
  queryEmbedding?: Float32Array;
  /** Detailed score breakdown per tool */
  scores: Record<string, number>;
  /** Human-readable explanation of why tools were selected (when debug: true) */
  explanation?: string;
  /** Retrieval latency in milliseconds */
  latencyMs: number;
}
