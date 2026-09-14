import { ToolImpulse } from '../core/engine.js';
import { RouterOptions, SessionState, ToolDefinition, ToolRouteResult } from '../core/types.js';

export interface AnthropicTool {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
  [key: string]: unknown;
}

export function anthropicToolToImpulseTool(tool: AnthropicTool): ToolDefinition {
  return {
    name: tool.name,
    description: tool.description || '',
    parameters: tool.input_schema,
  };
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
  engine: ToolImpulse,
  options?: RouterOptions
) {
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
      if (catalog.getToolNames().length === 0) {
        engine.registerToolsSync(allTools.map(anthropicToolToImpulseTool));
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
