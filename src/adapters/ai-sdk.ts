import { ToolImpulse } from '../core/engine.js';
import { RouterOptions, SessionState, ToolDefinition, ToolRouteResult } from '../core/types.js';
import { resolveScopedTools } from './utils.js';

export interface AiSdkToolRecord {
  [key: string]: {
    description?: string;
    parameters?: Record<string, unknown>;
    execute?: (...args: any[]) => Promise<any>;
    [key: string]: unknown;
  };
}

/**
 * Type-safe tool dictionary filter that preserves original tool types.
 */
export function filterTools<T extends Record<string, any>>(
  tools: T,
  selectedNames: string[]
): Partial<T> {
  const allowed = new Set(selectedNames);
  const filtered: Partial<T> = {};

  for (const [name, tool] of Object.entries(tools)) {
    if (allowed.has(name)) {
      filtered[name as keyof T] = tool;
    }
  }

  return filtered;
}

/**
 * Creates an in-memory tool router for Vercel AI SDK (`generateText` / `streamText`).
 *
 * @example
 * ```typescript
 * const router = createToolRouter(engine, { topK: 3 });
 * const { tools } = await router.getTools("Check Stripe invoices for Acme", allTools);
 *
 * const response = await generateText({
 *   model: openai('gpt-4o'),
 *   prompt: "Check Stripe invoices for Acme",
 *   tools,
 * });
 * ```
 */
export function createToolRouter<T extends AiSdkToolRecord>(
  engineOrOptions?: ToolImpulse | RouterOptions,
  maybeOptions?: RouterOptions
) {
  const engine = engineOrOptions instanceof ToolImpulse ? engineOrOptions : undefined;
  const options = engineOrOptions instanceof ToolImpulse ? maybeOptions : engineOrOptions;
  const toolsetCache = new WeakMap<object, ToolImpulse>();

  return {
    async getTools(
      queryOrMessages: string | Array<{ role: string; content: string }>,
      allTools: T,
      session?: SessionState
    ): Promise<{
      tools: Partial<T>;
      result: ToolRouteResult;
    }> {
      let query = '';
      if (typeof queryOrMessages === 'string') {
        query = queryOrMessages;
      } else if (Array.isArray(queryOrMessages)) {
        for (let i = queryOrMessages.length - 1; i >= 0; i--) {
          if (queryOrMessages[i].role === 'user') {
            query = queryOrMessages[i].content;
            break;
          }
        }
      }

      const toolDefs: ToolDefinition[] = Object.entries(allTools).map(([name, t]) => ({
        name,
        description: t.description || '',
        parameters: t.parameters,
      }));

      const result = await resolveScopedTools(
        engine,
        toolsetCache,
        allTools,
        toolDefs,
        query,
        session,
        options
      );

      const activeTools = filterTools(allTools, result.selectedNames);

      return {
        tools: activeTools,
        result,
      };
    },
  };
}

