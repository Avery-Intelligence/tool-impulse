import { ToolImpulse } from '../core/engine.js';
import { RouterOptions, SessionState, ToolDefinition, ToolRouteResult } from '../core/types.js';
import { resolveScopedTools } from './utils.js';

export interface AnthropicTool {
  name: string;
  description?: string;
  input_schema: {
    type: 'object';
    properties?: Record<string, unknown>;
    required?: string[];
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

/**
 * Convert an Anthropic tool definition into a ToolDefinition.
 */
export function fromAnthropicTool(tool: AnthropicTool): ToolDefinition {
  return {
    name: tool.name,
    description: tool.description || '',
    parameters: tool.input_schema,
  };
}

export function fromAnthropicTools(tools: AnthropicTool[]): ToolDefinition[] {
  return tools.map(fromAnthropicTool);
}

/**
 * Convert a ToolDefinition into an Anthropic tool.
 */
export function toAnthropicTool(tool: ToolDefinition): AnthropicTool {
  return {
    name: tool.name,
    description: tool.description,
    input_schema: {
      type: 'object',
      properties: (tool.parameters?.properties as Record<string, unknown>) || {},
      required: (tool.parameters?.required as string[]) || [],
    },
  };
}

export function toAnthropicTools(tools: ToolDefinition[]): AnthropicTool[] {
  return tools.map(toAnthropicTool);
}

/**
 * Creates an in-memory tool router for Anthropic Claude (`@anthropic-ai/sdk`).
 *
 * @example
 * ```typescript
 * import Anthropic from '@anthropic-ai/sdk';
 * import { ToolImpulse, createAnthropicToolFilter } from 'tool-impulse';
 *
 * const anthropic = new Anthropic();
 * const engine = new ToolImpulse();
 * const claudeFilter = createAnthropicToolFilter(engine, { topK: 3 });
 *
 * const { tools } = await claudeFilter.filterTools("Check pending customer invoices", allTools);
 *
 * const response = await anthropic.messages.create({
 *   model: 'claude-3-7-sonnet-20250219',
 *   max_tokens: 1024,
 *   messages: [{ role: 'user', content: "Check pending customer invoices" }],
 *   tools,
 * });
 * ```
 */
export function createAnthropicToolFilter(
  engineOrOptions?: ToolImpulse | RouterOptions,
  maybeOptions?: RouterOptions
) {
  const engine = engineOrOptions instanceof ToolImpulse ? engineOrOptions : undefined;
  const options = engineOrOptions instanceof ToolImpulse ? maybeOptions : engineOrOptions;
  const toolsetCache = new WeakMap<object, ToolImpulse>();

  return {
    async filterTools(
      query: string,
      allTools: AnthropicTool[],
      session?: SessionState
    ): Promise<{
      tools: AnthropicTool[];
      result: ToolRouteResult;
    }> {
      const toolDefs = fromAnthropicTools(allTools);
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
  };
}

// Convenient alias
export { createAnthropicToolFilter as createClaudeToolFilter };
