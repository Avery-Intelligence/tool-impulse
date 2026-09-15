import { ToolImpulse } from '../core/engine.js';
import { RouterOptions, SessionState, ToolDefinition, ToolRouteResult } from '../core/types.js';
import { resolveScopedTools } from './utils.js';

export interface McpToolDefinition {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
  [key: string]: unknown;
}

export function fromMCPTool(tool: McpToolDefinition): ToolDefinition {
  return {
    name: tool.name,
    description: tool.description || '',
    parameters: tool.inputSchema,
  };
}

export function fromMCPTools(tools: McpToolDefinition[]): ToolDefinition[] {
  return tools.map(fromMCPTool);
}

export function toMCPTool(tool: ToolDefinition): McpToolDefinition {
  return {
    name: tool.name,
    description: tool.description,
    inputSchema: tool.parameters,
  };
}

export function toMCPTools(tools: ToolDefinition[]): McpToolDefinition[] {
  return tools.map(toMCPTool);
}

export function createMcpFilter(engine: ToolImpulse, options?: RouterOptions) {
  const toolsetCache = new WeakMap<object, ToolImpulse>();

  return {
    /**
     * Filter a full list of MCP tool definitions down to the active turn subset.
     */
    async filterTools(
      query: string,
      allTools: McpToolDefinition[],
      session?: SessionState
    ): Promise<{
      tools: McpToolDefinition[];
      result: ToolRouteResult;
    }> {
      const toolDefs = fromMCPTools(allTools);
      const result = await resolveScopedTools(
        engine,
        toolsetCache,
        allTools,
        toolDefs,
        query,
        session,
        options
      );

      const selectedNames = new Set(result.selectedNames);
      const filtered = allTools.filter((t) => selectedNames.has(t.name));

      return {
        tools: filtered,
        result,
      };
    },
  };
}

