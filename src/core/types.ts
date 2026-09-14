/**
 * Tool Impulse Engine - Core Types
 *
 * Unconscious Perceptual Reflex for Autonomous Agent Tool Retrieval
 */

export type ToolVerbArchetype = 'read' | 'create' | 'update' | 'delete' | 'unknown';

export interface ImpulseTool {
  /** Unique tool name (e.g. "stripe_list_invoices", "jira_update_issue") */
  name: string;
  /** Natural language description of tool affordance */
  description: string;
  /** Optional JSON Schema parameter specification */
  parameters?: Record<string, unknown>;
  /** Functional family grouping (e.g. "stripe", "jira", "slack", "code") */
  family?: string;
  /** Domain fast-boost keywords */
  keywords?: string[];
  /** Is this tool idempotent / read-only */
  readOnly?: boolean;
}

export interface ImpulseSessionState {
  /** Embedding of previous turn query for trajectory blending */
  priorTurnEmbedding?: Float32Array | number[];
  /** Tool names recently executed in earlier turns (hysteresis bonus) */
  recentToolNames?: string[];
  /** Turn index in the conversation */
  turnCount?: number;
  /** Session ID / context tracking */
  sessionId?: string;
}

export interface ToolTransitionEdge {
  fromTool: string;
  toTool: string;
  weight: number;
}

export interface CodeTopologyProvider {
  /** Extract source code symbols from user query (functions, files, classes) */
  extractSymbols(query: string): string[];
  /** Return tool names strongly associated with the given code symbols */
  getRelatedTools(symbols: string[]): string[];
}

export interface ImpulseEmbedder {
  /** Vector dimension (e.g. 1536 for text-embedding-3-small) */
  readonly dimension: number;
  /** Compute embedding for single query string */
  embedQuery(text: string): Promise<Float32Array>;
  /** Compute embeddings for batch of texts (e.g. during tool registration) */
  embedBatch(texts: string[]): Promise<Float32Array[]>;
}

export interface ImpulseOptions {
  /** Maximum number of dynamic impulse tools to mount (default: 3) */
  topK?: number;
  /** Maximum tools allowed from a single family to enforce submodular diversity (default: 2) */
  maxPerFamily?: number;
  /** Dense vector vs sparse lexical weight: alpha * dense + (1 - alpha) * lexical (default: 0.70) */
  alpha?: number;
  /** Trajectory context blend factor: beta * current + (1 - beta) * prior (default: 0.75) */
  beta?: number;
  /** Hysteresis inertia bonus for tools invoked in recent turns (default: 0.25) */
  delta?: number;
  /** Spreading activation diffusion coefficient along companion graph edges (default: 0.18) */
  mu?: number;
  /** Optional code topology hook for developer tool boosting */
  codeTopology?: CodeTopologyProvider;
  /** Hard minimum score threshold for candidate selection (default: 0.10) */
  minScoreThreshold?: number;
}

export interface ScoredTool {
  tool: ImpulseTool;
  score: number;
  denseScore: number;
  sparseScore: number;
  activationBonus: number;
  hysteresisBonus: number;
  archetype: ToolVerbArchetype;
}

export interface ImpulseResult {
  /** Dynamic tools selected by the perceptual reflex (length <= topK) */
  tools: ImpulseTool[];
  /** All evaluated scores and metadata */
  scoredTools: ScoredTool[];
  /** The primary Top-1 anchor tool */
  primaryTool?: ImpulseTool;
  /** Companion tools mounted via spreading activation */
  companionTools: ImpulseTool[];
  /** Query embedding used for this turn (useful for chaining into next turn) */
  queryEmbedding?: Float32Array;
  /** Processing latency in milliseconds */
  latencyMs: number;
}
