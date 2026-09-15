import { ToolImpulse } from '../core/engine.js';
import { RouterOptions, SessionState, ToolDefinition, ToolRouteResult } from '../core/types.js';
import { resolveScopedTools } from './utils.js';

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

/**
 * Convert an OpenAI function tool definition into a ToolDefinition.
 */
export function fromOpenAITool(tool: OpenAiFunctionTool): ToolDefinition {
  return {
    name: tool.function.name,
    description: tool.function.description || '',
    parameters: tool.function.parameters,
  };
}

export function fromOpenAITools(tools: OpenAiFunctionTool[]): ToolDefinition[] {
  return tools.map(fromOpenAITool);
}

/**
 * Convert a ToolDefinition into an OpenAI function tool.
 */
export function toOpenAITool(tool: ToolDefinition): OpenAiFunctionTool {
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  };
}

export function toOpenAITools(tools: ToolDefinition[]): OpenAiFunctionTool[] {
  return tools.map(toOpenAITool);
}

/**
 * Creates an in-memory tool router for OpenAI, xAI Grok, Ollama, and OpenAI-compatible endpoints.
 *
 * @example
 * ```typescript
 * import OpenAI from 'openai';
 * import { ToolImpulse, createOpenAIToolFilter } from 'tool-impulse';
 *
 * const openai = new OpenAI();
 * const engine = new ToolImpulse();
 * const filter = createOpenAIToolFilter(engine, { topK: 3 });
 *
 * const { tools } = await filter.filterTools("Find critical bug reports in Jira", allTools);
 *
 * const response = await openai.chat.completions.create({
 *   model: 'gpt-5.6-sol',
 *   messages: [{ role: 'user', content: "Find critical bug reports in Jira" }],
 *   tools,
 * });
 * ```
 */
export function createOpenAIToolFilter(
  engineOrOptions?: ToolImpulse | RouterOptions,
  maybeOptions?: RouterOptions
) {
  const engine = engineOrOptions instanceof ToolImpulse ? engineOrOptions : undefined;
  const options = engineOrOptions instanceof ToolImpulse ? maybeOptions : engineOrOptions;

  return {
    async filterTools(
      query: string,
      allTools: OpenAiFunctionTool[],
      session?: SessionState
    ): Promise<{
      tools: OpenAiFunctionTool[];
      result: ToolRouteResult;
    }> {
      const toolDefs = fromOpenAITools(allTools);
      const result = await resolveScopedTools(
        engine,
        toolDefs,
        query,
        session,
        options
      );

      const allowed = new Set(result.selectedNames);
      const filtered = allTools.filter((t) => allowed.has(t.function.name));

      return {
        tools: filtered,
        result,
      };
    },
  };
}

export const createOpenAiToolFilter = createOpenAIToolFilter;
export const createGrokToolFilter = createOpenAIToolFilter;


