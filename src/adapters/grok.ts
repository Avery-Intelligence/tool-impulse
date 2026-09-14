import { ToolImpulse } from '../core/engine.js';
import { RouterOptions, SessionState, ToolDefinition, ToolRouteResult } from '../core/types.js';

export interface OpenAiFunctionTool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
    strict?: boolean;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export function grokToolToImpulseTool(tool: OpenAiFunctionTool): ToolDefinition {
  return {
    name: tool.function.name,
    description: tool.function.description || '',
    parameters: tool.function.parameters,
  };
}

/**
 * Creates an in-memory tool router for xAI Grok (`grok-2`, `grok-3`) and OpenAI-compatible endpoints.
 *
 * @example
 * ```typescript
 * import OpenAI from 'openai';
 * import { ToolImpulse, createGrokToolFilter } from 'tool-impulse';
 *
 * const xai = new OpenAI({
 *   apiKey: process.env.XAI_API_KEY,
 *   baseURL: 'https://api.x.ai/v1',
 * });
 *
 * const engine = new ToolImpulse();
 * const grokFilter = createGrokToolFilter(engine, { topK: 3 });
 *
 * // Filters 50+ tools down to top 3 active tools
 * const { tools } = await grokFilter.filterTools("Find recent critical bug reports in Jira", allTools);
 *
 * const response = await xai.chat.completions.create({
 *   model: 'grok-2',
 *   messages: [{ role: 'user', content: "Find recent critical bug reports in Jira" }],
 *   tools,
 * });
 * ```
 */
export function createGrokToolFilter(
  engine: ToolImpulse,
  options?: RouterOptions
) {
  return {
    async filterTools(
      query: string,
      allTools: OpenAiFunctionTool[],
      session?: SessionState
    ): Promise<{
      tools: OpenAiFunctionTool[];
      result: ToolRouteResult;
    }> {
      const catalog = engine.getCatalog();
      if (catalog.getToolNames().length === 0) {
        engine.registerToolsSync(allTools.map(grokToolToImpulseTool));
      }

      const result = await engine.resolve(query, session, options);
      const allowed = new Set(result.selectedNames);
      const filtered = allTools.filter((t) => allowed.has(t.function.name));

      return {
        tools: filtered,
        result,
      };
    },
  };
}

// Convenient aliases
export { createGrokToolFilter as createOpenAiToolFilter };
export { createGrokToolFilter as createOpenAiCompatibleFilter };
