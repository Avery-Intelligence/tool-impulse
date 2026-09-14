import { ToolImpulse } from '../core/engine.js';
import { RouterOptions, SessionState, ToolDefinition, ToolRouteResult } from '../core/types.js';

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
      const catalog = engine.getCatalog();
      const incomingNames = allTools.map((t) => t.name);
      const currentNames = catalog.getToolNames();

      const isStale =
        currentNames.length !== incomingNames.length ||
        incomingNames.some((name) => !catalog.hasTool(name));

      if (isStale) {
        if (engine.hasEmbedder()) {
          await engine.setTools(fromMCPTools(allTools));
        } else {
          engine.setToolsSync(fromMCPTools(allTools));
        }
      }

      const result = await engine.resolve(query, session, options);
      const selectedNames = new Set(result.selectedNames);
      const filtered = allTools.filter((t) => selectedNames.has(t.name));

      return {
        tools: filtered,
        result,
      };
    },
  };
}
