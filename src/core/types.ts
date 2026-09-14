/**
 * Tool Impulse Engine - Core Types
 *
 * Lightweight, in-memory tool retrieval and context optimization for LLM agents.
 */

export type ToolVerbKind = 'read' | 'create' | 'update' | 'delete' | 'other';

export interface ImpulseTool {
  /** Unique tool name (e.g. "stripe_list_invoices", "jira_update_issue") */
  name: string;
  /** Clear natural language description of what the tool does */
  description: string;
  /** Optional JSON Schema parameter specification */
  parameters?: Record<string, unknown>;
  /** Functional grouping / domain (e.g. "stripe", "jira", "slack", "database") */
  family?: string;
  /** Optional domain keywords for additional lexical boost */
  keywords?: string[];
  /** Flag indicating the tool is idempotent and safe to read */
  readOnly?: boolean;
}

export interface ImpulseSessionState {
  /** Embedding of the previous user turn to preserve context across pronoun shifts */
  priorTurnEmbedding?: Float32Array | number[];
  /** Names of tools recently executed in earlier turns for hysteresis inertia */
  recentToolNames?: string[];
  /** Current turn index in the multi-turn session */
  turnCount?: number;
  /** Optional session identifier */
  sessionId?: string;
}

export interface ToolTransitionEdge {
  fromTool: string;
  toTool: string;
  weight: number;
}

export interface CodeTopologyProvider {
  /** Detect code symbols (e.g. function names, file paths) in a query */
  extractSymbols(query: string): string[];
  /** Map detected code symbols to tool names that should be boosted */
  getRelatedTools(symbols: string[]): string[];
}

export interface ImpulseEmbedder {
  /** Vector dimension (e.g. 1536 for OpenAI text-embedding-3-small) */
  readonly dimension: number;
  /** Generate embedding for a single text string */
  embedQuery(text: string): Promise<Float32Array>;
  /** Generate embeddings for a batch of strings */
  embedBatch(texts: string[]): Promise<Float32Array[]>;
}

export interface ImpulseOptions {
  /** Maximum number of active tools to return (default: 3) */
  topK?: number;
  /** Maximum tools allowed from a single family to ensure diversity (default: 2) */
  maxPerFamily?: number;
  /** Weight for dense vector search vs sparse BM25: alpha * dense + (1 - alpha) * BM25 (default: 0.70) */
  alpha?: number;
  /** Blend factor for multi-turn trajectory: beta * current + (1 - beta) * prior (default: 0.75) */
  beta?: number;
  /** Inertia score bonus for tools executed in recent turns (default: 0.25) */
  inertiaBonus?: number;
  /** Graph diffusion boost for companion tools often used alongside the top match (default: 0.20) */
  companionBoost?: number;
  /** Optional code topology hook for boosting developer tools */
  codeTopology?: CodeTopologyProvider;
  /** Minimum score threshold for candidate selection (default: 0.05) */
  minScoreThreshold?: number;
}

export interface ScoredTool {
  tool: ImpulseTool;
  score: number;
  denseScore: number;
  bm25Score: number;
  companionBonus: number;
  inertiaBonus: number;
  verbKind: ToolVerbKind;
}

export interface ImpulseResult {
  /** The filtered active tools mounted for this turn (length <= topK) */
  tools: ImpulseTool[];
  /** Detailed score breakdown for all evaluated candidate tools */
  scoredTools: ScoredTool[];
  /** The top-1 anchor tool */
  primaryTool?: ImpulseTool;
  /** Companion tools mounted via graph co-occurrence */
  companionTools: ImpulseTool[];
  /** Contextualized query embedding for this turn (pass to next turn's session) */
  queryEmbedding?: Float32Array;
  /** Total retrieval latency in milliseconds */
  latencyMs: number;
}
