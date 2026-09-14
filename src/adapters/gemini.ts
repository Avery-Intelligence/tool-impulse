import { ToolImpulse } from '../core/engine.js';
import { RouterOptions, SessionState, ToolDefinition, ToolRouteResult } from '../core/types.js';

export interface GeminiFunctionDeclaration {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface GeminiToolGroup {
  functionDeclarations?: GeminiFunctionDeclaration[];
  [key: string]: unknown;
}

/**
 * Convert a Gemini function declaration into a ToolImpulse definition.
 */
export function fromGeminiDeclaration(decl: GeminiFunctionDeclaration): ToolDefinition {
  return {
    name: decl.name,
    description: decl.description || '',
    parameters: decl.parameters,
  };
}

export function fromGeminiDeclarations(decls: GeminiFunctionDeclaration[]): ToolDefinition[] {
  return decls.map(fromGeminiDeclaration);
}

/**
 * Convert a ToolImpulse definition into a Gemini FunctionDeclaration.
 */
export function toGeminiDeclaration(tool: ToolDefinition): GeminiFunctionDeclaration {
  return {
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
  };
}

export function toGeminiDeclarations(tools: ToolDefinition[]): GeminiFunctionDeclaration[] {
  return tools.map(toGeminiDeclaration);
}

/**
 * Creates an in-memory tool router for Google Gemini (`@google/genai`, `@google/generative-ai`, or Vertex AI).
 *
 * @example
 * ```typescript
 * import { GoogleGenAI } from '@google/genai';
 * import { ToolImpulse, createGeminiToolFilter } from 'tool-impulse';
 *
 * const engine = new ToolImpulse();
 * const geminiFilter = createGeminiToolFilter(engine, { topK: 3 });
 *
 * // Automatically formats { tools: [{ functionDeclarations: [...] }] } for Gemini
 * const { tools } = await geminiFilter.formatTools("Check customer balance in Stripe", allFunctionDeclarations);
 *
 * const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
 * const response = await ai.models.generateContent({
 *   model: 'gemini-2.5-flash',
 *   contents: "Check customer balance in Stripe",
 *   config: { tools },
 * });
 * ```
 */
export function createGeminiToolFilter(
  engineOrOptions?: ToolImpulse | RouterOptions,
  maybeOptions?: RouterOptions
) {
  const engine = engineOrOptions instanceof ToolImpulse ? engineOrOptions : new ToolImpulse();
  const options = engineOrOptions instanceof ToolImpulse ? maybeOptions : engineOrOptions;

  return {
    /**
     * Filter a list of Gemini function declarations down to the active turn subset.
     */
    async filterDeclarations(
      query: string,
      declarations: GeminiFunctionDeclaration[],
      session?: SessionState
    ): Promise<{
      declarations: GeminiFunctionDeclaration[];
      result: ToolRouteResult;
    }> {
      const catalog = engine.getCatalog();
      const incomingNames = declarations.map((d) => d.name);
      const currentNames = catalog.getToolNames();

      const isStale =
        currentNames.length !== incomingNames.length ||
        incomingNames.some((name) => !catalog.hasTool(name));

      if (isStale) {
        engine.setToolsSync(fromGeminiDeclarations(declarations));
      }

      const result = await engine.resolve(query, session, options);
      const allowed = new Set(result.selectedNames);
      const filtered = declarations.filter((d) => allowed.has(d.name));

      return {
        declarations: filtered,
        result,
      };
    },

    /**
     * Filter declarations and return the exact `{ tools: [{ functionDeclarations: [...] }] }`
     * shape expected by `@google/genai` and `@google/generative-ai`.
     */
    async formatTools(
      query: string,
      declarations: GeminiFunctionDeclaration[],
      session?: SessionState
    ): Promise<{
      tools: [{ functionDeclarations: GeminiFunctionDeclaration[] }];
      result: ToolRouteResult;
    }> {
      const { declarations: filtered, result } = await this.filterDeclarations(
        query,
        declarations,
        session
      );

      return {
        tools: [{ functionDeclarations: filtered }],
        result,
      };
    },

    /**
     * Filter pre-grouped Gemini tool arrays (e.g. `[{ functionDeclarations: [...] }]`).
     */
    async filterTools(
      query: string,
      toolGroups: GeminiToolGroup[],
      session?: SessionState
    ): Promise<{
      tools: GeminiToolGroup[];
      result: ToolRouteResult;
    }> {
      const allDecls: GeminiFunctionDeclaration[] = [];
      for (const group of toolGroups) {
        if (group.functionDeclarations) {
          allDecls.push(...group.functionDeclarations);
        }
      }

      const { declarations: filtered, result } = await this.filterDeclarations(
        query,
        allDecls,
        session
      );

      const filteredNames = new Set(filtered.map((d) => d.name));
      const outputGroups: GeminiToolGroup[] = [];

      for (const group of toolGroups) {
        if (!group.functionDeclarations) {
          outputGroups.push(group);
          continue;
        }

        const retained = group.functionDeclarations.filter((d) => filteredNames.has(d.name));
        if (retained.length > 0) {
          outputGroups.push({
            ...group,
            functionDeclarations: retained,
          });
        }
      }

      return {
        tools: outputGroups,
        result,
      };
    },
  };
}

// Convenient aliases
export { createGeminiToolFilter as createGeminiFilter };
