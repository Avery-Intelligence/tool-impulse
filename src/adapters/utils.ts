import { ToolImpulse } from '../core/engine.js';
import { RouterOptions, SessionState, ToolDefinition, ToolRouteResult } from '../core/types.js';

const LRU_MAX_SIZE = 128;
const catalogCache = new Map<string, ToolImpulse>();
const initPromises = new Map<string, Promise<ToolImpulse>>();
const baseInitPromises = new WeakMap<ToolImpulse, Promise<ToolImpulse>>();

/**
 * Computes a fast deterministic fingerprint for a tool catalog based on tool names and descriptions.
 * This guarantees that callers passing inline object literals (e.g. `{ stripe_refund: tool(...) }`)
 * hit the cache on subsequent requests instead of forcing a full index rebuild.
 */
export function getCatalogFingerprint(toolDefs: Array<{ name: string; description?: string; domain?: string }>): string {
  const sorted = [...toolDefs].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  let hash = 2166136261;
  for (let i = 0; i < sorted.length; i++) {
    const item = sorted[i];
    const str = `${item.name}:${item.description || ''}:${item.domain || ''}`;
    for (let j = 0; j < str.length; j++) {
      hash ^= str.charCodeAt(j);
      hash = Math.imul(hash, 16777619);
    }
  }
  return (hash >>> 0).toString(36) + '_' + sorted.length;
}

/**
 * Resolves active tools against either a pre-configured base engine or an isolated,
 * cached scoped engine.
 */
export async function resolveScopedTools(
  baseEngine: ToolImpulse | undefined,
  arg1: unknown,
  arg2?: unknown,
  arg3?: unknown,
  arg4?: unknown,
  arg5?: unknown,
  arg6?: unknown
): Promise<ToolRouteResult> {
  let toolDefs: ToolDefinition[];
  let query: string;
  let session: SessionState | undefined;
  let options: RouterOptions | undefined;

  if (Array.isArray(arg1)) {
    toolDefs = arg1 as ToolDefinition[];
    query = arg2 as string;
    session = arg3 as SessionState | undefined;
    options = arg4 as RouterOptions | undefined;
  } else {
    toolDefs = arg3 as ToolDefinition[];
    query = arg4 as string;
    session = arg5 as SessionState | undefined;
    options = arg6 as RouterOptions | undefined;
  }

  const incomingNames = toolDefs.map((t) => t.name);

  // 1. If baseEngine was initialized with these exact tools, reuse it directly (read-only)
  if (baseEngine) {
    const baseCatalog = baseEngine.getCatalog();
    const baseNames = baseCatalog.getToolNames();
    if (
      baseNames.length === incomingNames.length &&
      incomingNames.every((name) => baseCatalog.hasTool(name))
    ) {
      return baseEngine.resolve(query, session, options);
    }

    // If baseEngine is empty/uninitialized, initialize it directly for the base tenant
    if (baseNames.length === 0) {
      let initPromise = baseInitPromises.get(baseEngine);
      if (!initPromise) {
        initPromise = (async () => {
          if (baseEngine.hasEmbedder()) {
            await baseEngine.setTools(toolDefs);
          } else {
            baseEngine.setToolsSync(toolDefs);
          }
          return baseEngine;
        })();
        baseInitPromises.set(baseEngine, initPromise);
      }
      await initPromise;
      return baseEngine.resolve(query, session, options);
    }
  }

  // 2. Lookup in bounded fingerprint LRU cache
  const fingerprint = getCatalogFingerprint(toolDefs);
  let scopedEngine = catalogCache.get(fingerprint);

  if (scopedEngine) {
    // Refresh LRU order
    catalogCache.delete(fingerprint);
    catalogCache.set(fingerprint, scopedEngine);
    return scopedEngine.resolve(query, session, options);
  }

  // 3. In-flight deduplication if concurrent requests arrive for a new catalog
  let inFlight = initPromises.get(fingerprint);
  if (!inFlight) {
    inFlight = (async () => {
      try {
        const embedder = baseEngine?.getEmbedder();
        const instance = new ToolImpulse({
          tools: toolDefs,
          embedder,
          defaultOptions: baseEngine?.getDefaultOptions(),
        });
        if (embedder) {
          await instance.setTools(toolDefs);
        }

        // Evict oldest entry if at capacity
        if (catalogCache.size >= LRU_MAX_SIZE) {
          const oldestKey = catalogCache.keys().next().value;
          if (oldestKey) catalogCache.delete(oldestKey);
        }
        catalogCache.set(fingerprint, instance);
        return instance;
      } finally {
        initPromises.delete(fingerprint);
      }
    })();
    initPromises.set(fingerprint, inFlight);
  }

  scopedEngine = await inFlight;
  return scopedEngine.resolve(query, session, options);
}
