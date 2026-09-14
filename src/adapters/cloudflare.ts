import { ToolImpulse } from '../core/engine.js';
import { RouterOptions, SessionState, ToolDefinition, ToolRouteResult } from '../core/types.js';

export interface CloudflareAiTool {
  name: string;
  description: string;
  parameters?: Record<string, unknown>;
  [key: string]: unknown;
}

export function fromCloudflareAiTool(tool: CloudflareAiTool): ToolDefinition {
  return {
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
  };
}

export function fromCloudflareAiTools(tools: CloudflareAiTool[]): ToolDefinition[] {
  return tools.map(fromCloudflareAiTool);
}

export function toCloudflareAiTool(tool: ToolDefinition): CloudflareAiTool {
  return {
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
  };
}

export function toCloudflareAiTools(tools: ToolDefinition[]): CloudflareAiTool[] {
  return tools.map(toCloudflareAiTool);
}

/**
 * Creates an in-memory tool router for Cloudflare Workers and Workers AI.
 *
 * Runs inside V8 Isolates with zero dependencies and <0.1ms CPU overhead.
 *
 * @example
 * ```typescript
 * import { ToolImpulse, createCloudflareAiFilter } from 'tool-impulse';
 *
 * export default {
 *   async fetch(request: Request, env: Env) {
 *     const engine = new ToolImpulse();
 *     const cfFilter = createCloudflareAiFilter(engine, { topK: 3 });
 *
 *     const { query } = await request.json();
 *     const { tools } = await cfFilter.filterTools(query, myCloudflareTools);
 *
 *     const response = await env.AI.run('@cf/meta/llama-3-8b-instruct', {
 *       prompt: query,
 *       tools,
 *     });
 *     return Response.json(response);
 *   }
 * };
 * ```
 */
export function createCloudflareAiFilter(
  engine: ToolImpulse,
  options?: RouterOptions
) {
  return {
    async filterTools<T extends CloudflareAiTool>(
      query: string,
      allTools: T[],
      session?: SessionState
    ): Promise<{
      tools: T[];
      result: ToolRouteResult;
    }> {
      if (engine.getCatalog().getToolNames().length === 0) {
        engine.registerToolsSync(fromCloudflareAiTools(allTools));
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
