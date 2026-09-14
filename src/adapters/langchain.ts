import { ToolImpulseEngine } from '../core/engine.js';
import { ImpulseOptions, ImpulseResult, ImpulseSessionState, ImpulseTool } from '../core/types.js';

export interface LangChainToolLike {
  name: string;
  description: string;
  schema?: Record<string, unknown>;
  call?: (...args: any[]) => Promise<any>;
  [key: string]: unknown;
}

export function langChainToImpulseTool(tool: LangChainToolLike): ImpulseTool {
  return {
    name: tool.name,
    description: tool.description,
    parameters: tool.schema,
  };
}

/**
 * Creates a LangChain / LangGraph compatible dynamic tool retriever.
 * Allows agents to project a subset of relevant tools based on user input or state.
 *
 * @example
 * ```typescript
 * const retriever = createLangChainImpulseRetriever(engine, myAllTools, { topK: 3 });
 * const activeTools = await retriever.getTools("Check Stripe invoices");
 * ```
 */
export function createLangChainImpulseRetriever<T extends LangChainToolLike>(
  engine: ToolImpulseEngine,
  allTools: T[],
  options?: ImpulseOptions
) {
  // Auto-register tools on creation if engine bank is empty
  const bank = engine.getBank();
  if (bank.getToolNames().length === 0) {
    const impulseTools = allTools.map(langChainToImpulseTool);
    engine.registerToolsSync(impulseTools);
  }

  return {
    /**
     * Retrieve the top-K active tools for a given user query or agent thought.
     */
    async getTools(
      query: string,
      session?: ImpulseSessionState
    ): Promise<{
      tools: T[];
      impulseResult: ImpulseResult;
    }> {
      const impulseResult = await engine.resolve(query, session, options);
      const selectedNames = new Set(impulseResult.tools.map((t) => t.name));

      const filtered = allTools.filter((t) => selectedNames.has(t.name));

      return {
        tools: filtered,
        impulseResult,
      };
    },

    /**
     * Filter an arbitrary subset of tools dynamically.
     */
    async filterTools(
      query: string,
      candidateTools: T[],
      session?: ImpulseSessionState
    ): Promise<T[]> {
      const impulseResult = await engine.resolve(query, session, options);
      const selectedNames = new Set(impulseResult.tools.map((t) => t.name));

      return candidateTools.filter((t) => selectedNames.has(t.name));
    },
  };
}
