import { ToolImpulse } from '../core/engine.js';
import { RouterOptions, SessionState, ToolDefinition, ToolRouteResult } from '../core/types.js';

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
 * Convert an Anthropic tool definition into a ToolImpulse definition.
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
 * Convert a ToolImpulse definition into an Anthropic tool definition.
 */
export function toAnthropicTool(tool: ToolDefinition): AnthropicTool {
  return {
    name: tool.name,
    description: tool.description,
    input_schema: (tool.parameters as AnthropicTool['input_schema']) || {
      type: 'object',
      properties: {},
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
 *   model: 'claude-3-5-sonnet-20241022',
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
  const engine = engineOrOptions instanceof ToolImpulse ? engineOrOptions : new ToolImpulse();
  const options = engineOrOptions instanceof ToolImpulse ? maybeOptions : engineOrOptions;

  return {
    async filterTools(
      query: string,
      allTools: AnthropicTool[],
      session?: SessionState
    ): Promise<{
      tools: AnthropicTool[];
      result: ToolRouteResult;
    }> {
      const catalog = engine.getCatalog();
      const incomingNames = allTools.map((t) => t.name);
      const currentNames = catalog.getToolNames();

      const isStale =
        currentNames.length !== incomingNames.length ||
        incomingNames.some((name) => !catalog.hasTool(name));

      if (isStale) {
        engine.setToolsSync(fromAnthropicTools(allTools));
      }

      const result = await engine.resolve(query, session, options);
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
