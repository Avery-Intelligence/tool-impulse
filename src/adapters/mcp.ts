import { ToolImpulseEngine } from '../core/engine.js';
import { ImpulseOptions, ImpulseResult, ImpulseSessionState, ImpulseTool } from '../core/types.js';

export interface McpToolDefinition {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
  [key: string]: unknown;
}

export function mcpToImpulseTool(tool: McpToolDefinition): ImpulseTool {
  return {
    name: tool.name,
    description: tool.description || '',
    parameters: tool.inputSchema,
  };
}

export function impulseToMcpTool(tool: ImpulseTool): McpToolDefinition {
  return {
    name: tool.name,
    description: tool.description,
    inputSchema: tool.parameters,
  };
}

export function createMcpFilter(engine: ToolImpulseEngine, options?: ImpulseOptions) {
  return {
    /**
     * Filter a full list of MCP tool definitions down to the dynamically active impulse tools.
     */
    async filterTools(
      query: string,
      allTools: McpToolDefinition[],
      session?: ImpulseSessionState
    ): Promise<{
      tools: McpToolDefinition[];
      impulseResult: ImpulseResult;
    }> {
      const bank = engine.getBank();
      if (bank.getToolNames().length === 0) {
        const impulseTools = allTools.map(mcpToImpulseTool);
        engine.registerToolsSync(impulseTools);
      }

      const impulseResult = await engine.resolve(query, session, options);
      const selectedNames = new Set(impulseResult.tools.map((t) => t.name));

      const filtered = allTools.filter((t) => selectedNames.has(t.name));

      return {
        tools: filtered,
        impulseResult,
      };
    },
  };
}
