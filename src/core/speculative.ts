import { ImpulseResult, ImpulseTool } from './types.js';

export interface SpeculativeExecutor<T = any> {
  (tool: ImpulseTool, query: string): Promise<T>;
}

export interface SpeculativeExecutionResult<T = any> {
  tool: ImpulseTool;
  promise?: Promise<T>;
  status: 'dispatched' | 'skipped';
  reason?: string;
}

export interface SpeculativeOptions {
  /** Minimum score required to trigger speculative execution (default: 0.85) */
  confidenceThreshold?: number;
  /** Require tool to be explicitly marked readOnly (default: false, allows archetype read) */
  strictReadOnly?: boolean;
}

/**
 * Speculative Pre-Execution Helper (Whitepaper Section 3.7)
 *
 * Dispatches idempotent read operations in parallel during the prompt construction window
 * when top-1 tool score exceeds the high-confidence threshold.
 */
export class SpeculativeRunner {
  public static maybeDispatch<T = any>(
    result: ImpulseResult,
    query: string,
    executor: SpeculativeExecutor<T>,
    options: SpeculativeOptions = {}
  ): SpeculativeExecutionResult<T> {
    const threshold = options.confidenceThreshold || 0.85;

    if (!result.primaryTool) {
      return { tool: result.tools[0], status: 'skipped', reason: 'No primary tool identified' };
    }

    const primaryTool = result.primaryTool;
    const scoredPrimary = result.scoredTools.find((s) => s.tool.name === primaryTool.name);

    if (!scoredPrimary || scoredPrimary.score < threshold) {
      return {
        tool: primaryTool,
        status: 'skipped',
        reason: `Score ${scoredPrimary?.score.toFixed(3) || 0} below threshold ${threshold}`,
      };
    }

    // Safety check: Idempotency
    const isIdempotent = primaryTool.readOnly || scoredPrimary.archetype === 'read';
    if (!isIdempotent) {
      return {
        tool: primaryTool,
        status: 'skipped',
        reason: 'Tool is not classified as an idempotent read operation',
      };
    }

    // Dispatch speculative execution in parallel
    const promise = executor(primaryTool, query).catch((err) => {
      // Catch error to prevent unhandled rejection during speculative pre-flight
      return { error: err.message, speculativeFailure: true } as any;
    });

    return {
      tool: primaryTool,
      promise,
      status: 'dispatched',
    };
  }
}
