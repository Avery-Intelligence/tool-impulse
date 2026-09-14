import { ImpulseBank } from './bank.js';
import {
  ImpulseOptions,
  ImpulseResult,
  ImpulseSessionState,
  ImpulseTool,
  ScoredTool,
  ToolVerbArchetype,
} from './types.js';

const READ_VERBS = new Set(['get', 'list', 'search', 'find', 'read', 'fetch', 'show', 'view', 'check', 'status', 'inspect', 'query', 'download', 'blast', 'deps', 'map', 'tour']);
const CREATE_VERBS = new Set(['create', 'add', 'insert', 'new', 'post', 'generate', 'send', 'upload', 'clone', 'register', 'provision', 'start', 'begin', 'claim']);
const UPDATE_VERBS = new Set(['update', 'edit', 'modify', 'patch', 'put', 'change', 'set', 'rename', 'archive', 'close', 'reopen', 'assign', 'resolve', 'transition']);
const DELETE_VERBS = new Set(['delete', 'remove', 'drop', 'destroy', 'cancel', 'erase', 'kill', 'purge', 'uninstall']);

export class ImpulseResolver {
  private bank: ImpulseBank;
  private defaultOptions: Required<Omit<ImpulseOptions, 'codeTopology'>> = {
    topK: 3,
    maxPerFamily: 2,
    alpha: 0.70,
    beta: 0.75,
    delta: 0.25,
    mu: 0.18,
    minScoreThreshold: 0.10,
  };

  constructor(bank: ImpulseBank) {
    this.bank = bank;
  }

  /**
   * Fast lexical tokenization splitting camelCase, snake_case, and non-alphanumerics.
   */
  public tokenize(text: string): string[] {
    return text
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((t) => t.length > 1);
  }

  /**
   * Classify tool verb archetype based on tool name and description.
   */
  public getToolArchetype(tool: ImpulseTool): ToolVerbArchetype {
    const tokens = this.tokenize(tool.name);
    for (const token of tokens) {
      if (READ_VERBS.has(token)) return 'read';
      if (CREATE_VERBS.has(token)) return 'create';
      if (UPDATE_VERBS.has(token)) return 'update';
      if (DELETE_VERBS.has(token)) return 'delete';
    }

    const descTokens = this.tokenize(tool.description).slice(0, 10);
    for (const token of descTokens) {
      if (READ_VERBS.has(token)) return 'read';
      if (CREATE_VERBS.has(token)) return 'create';
      if (UPDATE_VERBS.has(token)) return 'update';
      if (DELETE_VERBS.has(token)) return 'delete';
    }

    return 'unknown';
  }

  /**
   * Detect user query intent archetype.
   */
  public detectQueryIntent(query: string): ToolVerbArchetype {
    const tokens = this.tokenize(query);
    for (const token of tokens) {
      if (DELETE_VERBS.has(token)) return 'delete';
      if (UPDATE_VERBS.has(token)) return 'update';
      if (CREATE_VERBS.has(token)) return 'create';
      if (READ_VERBS.has(token)) return 'read';
    }
    return 'read'; // Default safe assumption is exploratory read
  }

  /**
   * Fast lexical BM25-like sparse scoring over tool name, description, and keywords.
   */
  public computeSparseScore(queryTokens: string[], tool: ImpulseTool): number {
    if (queryTokens.length === 0) return 0.0;

    const nameTokens = new Set(this.tokenize(tool.name));
    const descTokens = new Set(this.tokenize(tool.description));
    const keywords = new Set((tool.keywords || []).map((k) => k.toLowerCase()));

    let matches = 0;
    let nameMatches = 0;
    let keywordMatches = 0;

    for (const q of queryTokens) {
      if (nameTokens.has(q)) {
        matches += 2.0;
        nameMatches++;
      } else if (keywords.has(q)) {
        matches += 1.8;
        keywordMatches++;
      } else if (descTokens.has(q)) {
        matches += 1.0;
      }
    }

    // Normalized BM25 proxy bounded in [0, 1]
    const rawScore = matches / (queryTokens.length + 1.5);
    const boost = (nameMatches > 0 ? 0.25 : 0) + (keywordMatches > 0 ? 0.20 : 0);

    return Math.min(1.0, rawScore + boost);
  }

  /**
   * Contextualize query embedding with prior turn embedding (Trajectory Blending).
   */
  public contextualizeEmbedding(
    currentEmbedding: Float32Array,
    priorEmbedding?: Float32Array | number[],
    beta: number = 0.75
  ): Float32Array {
    if (!priorEmbedding) return currentEmbedding;

    const prior = priorEmbedding instanceof Float32Array ? priorEmbedding : new Float32Array(priorEmbedding);
    if (prior.length !== currentEmbedding.length) return currentEmbedding;

    const blended = new Float32Array(currentEmbedding.length);
    let sumSq = 0;

    for (let i = 0; i < blended.length; i++) {
      const val = beta * currentEmbedding[i] + (1 - beta) * prior[i];
      blended[i] = val;
      sumSq += val * val;
    }

    // Re-normalize to unit length
    const norm = Math.sqrt(sumSq) || 1e-10;
    for (let i = 0; i < blended.length; i++) {
      blended[i] /= norm;
    }

    return blended;
  }

  /**
   * Main Perceptual Reflex Resolver.
   * Runs in <0.08ms in-memory across registered tools.
   */
  public resolve(
    query: string,
    queryEmbedding?: Float32Array,
    session?: ImpulseSessionState,
    options?: ImpulseOptions
  ): ImpulseResult {
    const startTime = performance.now();
    const opts = { ...this.defaultOptions, ...options };
    const queryTokens = this.tokenize(query);
    const queryIntent = this.detectQueryIntent(query);

    // 1. Contextualize query embedding if prior turn exists
    let activeVec: Float32Array | undefined = queryEmbedding;
    if (queryEmbedding && session?.priorTurnEmbedding) {
      activeVec = this.contextualizeEmbedding(queryEmbedding, session.priorTurnEmbedding, opts.beta);
    }

    const recentSet = new Set(session?.recentToolNames || []);
    const scoredList: ScoredTool[] = [];
    const allTools = this.bank.getAllTools();

    // 2. Score all candidate tools
    for (const tool of allTools) {
      const dense = activeVec ? this.bank.computeCosine(activeVec, tool.name) : 0.0;
      const sparse = this.computeSparseScore(queryTokens, tool);
      const archetype = this.getToolArchetype(tool);

      // Hybrid combination
      const baseScore = activeVec ? opts.alpha * dense + (1 - opts.alpha) * sparse : sparse;

      // Hysteresis inertia bonus for tools active in recent turns
      const hysteresisBonus = recentSet.has(tool.name) ? opts.delta : 0.0;

      scoredList.push({
        tool,
        score: baseScore + hysteresisBonus,
        denseScore: dense,
        sparseScore: sparse,
        activationBonus: 0.0,
        hysteresisBonus,
        archetype,
      });
    }

    // 3. Apply Code Topology Boost if provider supplied
    if (options?.codeTopology) {
      const symbols = options.codeTopology.extractSymbols(query);
      if (symbols.length > 0) {
        const relatedTools = new Set(options.codeTopology.getRelatedTools(symbols));
        for (const item of scoredList) {
          if (relatedTools.has(item.tool.name)) {
            item.score += 0.30;
          }
        }
      }
    }

    // Sort by preliminary score descending
    scoredList.sort((a, b) => b.score - a.score);

    // 4. Identify Anchor Tool (Top-1) and Spreading Activation
    let primaryTool: ImpulseTool | undefined = undefined;
    const companionTools: ImpulseTool[] = [];

    if (scoredList.length > 0 && scoredList[0].score >= opts.minScoreThreshold) {
      const anchor = scoredList[0];
      primaryTool = anchor.tool;
      const anchorScore = anchor.score;

      // Diffuse activation along outgoing graph edges from anchor
      const neighbors = this.bank.getNeighbors(anchor.tool.name);
      if (neighbors.size > 0) {
        for (const candidate of scoredList) {
          if (candidate.tool.name === anchor.tool.name) continue;

          const edgeWeight = neighbors.get(candidate.tool.name);
          if (edgeWeight && edgeWeight > 0) {
            const bonus = opts.mu * edgeWeight;
            candidate.activationBonus = bonus;
            candidate.score += bonus;

            // ANCHOR PROTECTION RULE: Companions must NEVER leapfrog the primary anchor
            if (candidate.score >= anchorScore) {
              candidate.score = anchorScore - 0.001;
            }
            companionTools.push(candidate.tool);
          }
        }

        // Re-sort after spreading activation diffusion
        scoredList.sort((a, b) => b.score - a.score);
      }
    }

    // 5. Monotone Submodular Family Diversity Selection
    const selected: ImpulseTool[] = [];
    const familyCounts = new Map<string, number>();

    // Pass 1: Greedily pick top tools adhering to family caps and safety archetype constraints
    for (const candidate of scoredList) {
      if (selected.length >= opts.topK) break;
      if (candidate.score < opts.minScoreThreshold) continue;

      const family = candidate.tool.family || 'default';
      const count = familyCounts.get(family) || 0;

      if (count >= opts.maxPerFamily) {
        continue; // Enforce MAX_PER_FAMILY submodular cap
      }

      // Safety constraint: If user is performing read/query, deprioritize destructive deletes
      if (queryIntent === 'read' && candidate.archetype === 'delete' && candidate.score < 0.85) {
        continue;
      }

      selected.push(candidate.tool);
      familyCounts.set(family, count + 1);
    }

    // Pass 2: Graceful backfill if diversity constraints left slots unfilled
    if (selected.length < opts.topK) {
      const selectedNames = new Set(selected.map((t) => t.name));
      for (const candidate of scoredList) {
        if (selected.length >= opts.topK) break;
        if (candidate.score < opts.minScoreThreshold) continue;
        if (selectedNames.has(candidate.tool.name)) continue;

        selected.push(candidate.tool);
        selectedNames.add(candidate.tool.name);
      }
    }

    const latencyMs = performance.now() - startTime;

    return {
      tools: selected,
      scoredTools: scoredList,
      primaryTool,
      companionTools,
      queryEmbedding: activeVec,
      latencyMs,
    };
  }
}
