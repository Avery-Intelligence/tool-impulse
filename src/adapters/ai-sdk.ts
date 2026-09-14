import { ToolImpulseEngine } from '../core/engine.js';
import { ImpulseOptions, ImpulseResult, ImpulseSessionState, ImpulseTool } from '../core/types.js';

export interface AiSdkToolLike {
  description?: string;
  parameters?: Record<string, unknown>;
  execute?: (...args: any[]) => Promise<any>;
}

export interface ImpulseAiSdkFilterOptions extends ImpulseOptions {
  /** Extract user query from message history if query string is not provided directly */
  extractLastUserQuery?: boolean;
}

/**
 * Creates an intelligent per-turn tool filter for Vercel AI SDK (generateText / streamText).
 *
 * @example
 * ```typescript
 * const impulse = createAiSdkImpulseMiddleware(engine);
 * const activeTools = await impulse.getTools("Check Stripe invoices for acme", allTools);
 *
 * const result = await generateText({
 *   model: openai('gpt-4o'),
 *   messages,
 *   tools: activeTools.tools,
 * });
 * ```
 */
export function createAiSdkImpulseMiddleware<T extends Record<string, AiSdkToolLike>>(
  engine: ToolImpulseEngine,
  options?: ImpulseAiSdkFilterOptions
) {
  return {
    async getTools(
      queryOrMessages: string | Array<{ role: string; content: string }>,
      allTools: T,
      session?: ImpulseSessionState
    ): Promise<{
      tools: Partial<T>;
      impulseResult: ImpulseResult;
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

      // Check if allTools are registered in the engine; if bank is empty, register them
      const bank = engine.getBank();
      if (bank.getToolNames().length === 0) {
        const impulseTools: ImpulseTool[] = Object.entries(allTools).map(([name, tool]) => ({
          name,
          description: tool.description || '',
          parameters: tool.parameters,
        }));
        engine.registerToolsSync(impulseTools);
      }

      const impulseResult = await engine.resolve(query, session, options);
      const selectedNames = new Set(impulseResult.tools.map((t) => t.name));

      const filteredTools: Partial<T> = {};
      for (const [name, tool] of Object.entries(allTools)) {
        if (selectedNames.has(name)) {
          filteredTools[name as keyof T] = tool as any;
        }
      }

      return {
        tools: filteredTools,
        impulseResult,
      };
    },
  };
}
