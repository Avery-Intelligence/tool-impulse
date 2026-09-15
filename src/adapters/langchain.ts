import { ToolImpulse } from '../core/engine.js';
import { RouterOptions, SessionState, ToolDefinition, ToolRouteResult } from '../core/types.js';
import { resolveScopedTools } from './utils.js';

export interface LangChainToolLike {
  name: string;
  description: string;
  schema?: Record<string, unknown>;
  call?: (...args: unknown[]) => Promise<unknown> | unknown;
  [key: string]: unknown;
}

/**
 * Creates a dynamic tool retriever for LangChain / LangGraph agents.
 */
export function createLangChainRetriever<T extends LangChainToolLike>(
  engine: ToolImpulse,
  allTools: T[],
  options?: RouterOptions
) {
  const toolsetCache = new WeakMap<object, ToolImpulse>();

  return {
    async getTools(
      query: string,
      session?: SessionState
    ): Promise<{
      tools: T[];
      result: ToolRouteResult;
    }> {
      const toolDefs: ToolDefinition[] = allTools.map((t) => ({
        name: t.name,
        description: t.description,
        parameters: t.schema,
      }));

      const result = await resolveScopedTools(
        engine,
        toolsetCache,
        allTools,
        toolDefs,
        query,
        session,
        options
      );

      const allowed = new Set(result.selectedNames);
      const filtered = allTools.filter((t) => allowed.has(t.name));

      return {
        tools: filtered,
        result,
      };
    },

    async filterTools(
      query: string,
      candidateTools: T[],
      session?: SessionState
    ): Promise<T[]> {
      const toolDefs: ToolDefinition[] = candidateTools.map((t) => ({
        name: t.name,
        description: t.description,
        parameters: t.schema,
      }));

      const result = await resolveScopedTools(
        engine,
        toolsetCache,
        candidateTools,
        toolDefs,
        query,
        session,
        options
      );

      const allowed = new Set(result.selectedNames);
      return candidateTools.filter((t) => allowed.has(t.name));
    },
  };
}

