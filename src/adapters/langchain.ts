import { ToolImpulse } from '../core/engine.js';
import { RouterOptions, SessionState, ToolDefinition, ToolRouteResult } from '../core/types.js';

export interface LangChainToolLike {
  name: string;
  description: string;
  schema?: Record<string, unknown>;
  call?: (...args: any[]) => Promise<any>;
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
  const catalog = engine.getCatalog();
  if (catalog.getToolNames().length === 0) {
    const toolDefs: ToolDefinition[] = allTools.map((t) => ({
      name: t.name,
      description: t.description,
      parameters: t.schema,
    }));
    engine.registerToolsSync(toolDefs);
  }

  return {
    async getTools(
      query: string,
      session?: SessionState
    ): Promise<{
      tools: T[];
      result: ToolRouteResult;
    }> {
      const result = await engine.resolve(query, session, options);
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
      const result = await engine.resolve(query, session, options);
      const allowed = new Set(result.selectedNames);
      return candidateTools.filter((t) => allowed.has(t.name));
    },
  };
}

export { createLangChainRetriever as createLangChainImpulseRetriever };
