import { ImpulseBank } from './bank.js';
import { OkapiBM25 } from './bm25.js';
import {
  ImpulseOptions,
  ImpulseResult,
  ImpulseSessionState,
  ImpulseTool,
  ScoredTool,
  ToolVerbKind,
} from './types.js';

const READ_VERBS = new Set(['get', 'list', 'search', 'find', 'read', 'fetch', 'show', 'view', 'check', 'status', 'inspect', 'query', 'download']);
const CREATE_VERBS = new Set(['create', 'add', 'insert', 'new', 'post', 'generate', 'send', 'upload', 'clone', 'register', 'provision', 'start']);
const UPDATE_VERBS = new Set(['update', 'edit', 'modify', 'patch', 'put', 'change', 'set', 'rename', 'archive', 'close', 'reopen', 'assign', 'resolve']);
const DELETE_VERBS = new Set(['delete', 'remove', 'drop', 'destroy', 'cancel', 'erase', 'kill', 'purge', 'uninstall']);

export class ImpulseResolver {
  private bank: ImpulseBank;
  private bm25: OkapiBM25;
  private defaultOptions: Required<Omit<ImpulseOptions, 'codeTopology'>> = {
    topK: 3,
    maxPerFamily: 2,
    alpha: 0.70,
    beta: 0.75,
    inertiaBonus: 0.25,
    companionBoost: 0.20,
    minScoreThreshold: 0.05,
  };

  constructor(bank: ImpulseBank) {
    this.bank = bank;
    this.bm25 = new OkapiBM25();
  }

  /**
   * Re-index BM25 search corpus from bank tools.
   */
  public syncBm25Index(): void {
    const tools = this.bank.getAllTools();
    const docs = tools.map((t) => ({
      id: t.name,
      text: `${t.name} ${t.description} ${(t.keywords || []).join(' ')}`,
    }));
    this.bm25.indexDocuments(docs);
  }

  /**
   * Detect general verb kind for tool classification.
   */
  public getToolVerbKind(tool: ImpulseTool): ToolVerbKind {
    const tokens = OkapiBM25.tokenize(tool.name);
    for (const token of tokens) {
      if (READ_VERBS.has(token)) return 'read';
      if (CREATE_VERBS.has(token)) return 'create';
      if (UPDATE_VERBS.has(token)) return 'update';
      if (DELETE_VERBS.has(token)) return 'delete';
    }

    const descTokens = OkapiBM25.tokenize(tool.description).slice(0, 10);
    for (const token of descTokens) {
      if (READ_VERBS.has(token)) return 'read';
      if (CREATE_VERBS.has(token)) return 'create';
      if (UPDATE_VERBS.has(token)) return 'update';
      if (DELETE_VERBS.has(token)) return 'delete';
    }

    return 'other';
  }

  /**
   * Detect user query intent.
   */
  public detectQueryIntent(query: string): ToolVerbKind {
    const tokens = OkapiBM25.tokenize(query);
    for (const token of tokens) {
      if (DELETE_VERBS.has(token)) return 'delete';
      if (UPDATE_VERBS.has(token)) return 'update';
      if (CREATE_VERBS.has(token)) return 'create';
      if (READ_VERBS.has(token)) return 'read';
    }
    return 'read';
  }

  /**
   * Blend previous turn embedding with current query vector to resolve pronoun shifts.
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

    const norm = Math.sqrt(sumSq) || 1e-10;
    for (let i = 0; i < blended.length; i++) {
      blended[i] /= norm;
    }

    return blended;
  }

  /**
   * Resolve top-K tools for an incoming user query.
   */
  public resolve(
    query: string,
    queryEmbedding?: Float32Array,
    session?: ImpulseSessionState,
    options?: ImpulseOptions
  ): ImpulseResult {
    const startTime = performance.now();
    const opts = { ...this.defaultOptions, ...options };
    const queryIntent = this.detectQueryIntent(query);

    // 1. Contextualize query embedding if prior turn exists
    let activeVec: Float32Array | undefined = queryEmbedding;
    if (queryEmbedding && session?.priorTurnEmbedding) {
      activeVec = this.contextualizeEmbedding(queryEmbedding, session.priorTurnEmbedding, opts.beta);
    }

    // 2. Score via Okapi BM25 lexical index
    const bm25Scores = this.bm25.score(query);
    const recentSet = new Set(session?.recentToolNames || []);
    const scoredList: ScoredTool[] = [];
    const allTools = this.bank.getAllTools();

    // 3. Compute hybrid scores
    for (const tool of allTools) {
      const dense = activeVec ? this.bank.computeCosine(activeVec, tool.name) : 0.0;
      const sparse = bm25Scores.get(tool.name) || 0.0;
      const verbKind = this.getToolVerbKind(tool);

      // Weighted dense-sparse score
      const baseScore = activeVec ? opts.alpha * dense + (1 - opts.alpha) * sparse : sparse;

      // Hysteresis inertia bonus for tools active in recent turns
      const inertia = recentSet.has(tool.name) ? opts.inertiaBonus : 0.0;

      scoredList.push({
        tool,
        score: baseScore + inertia,
        denseScore: dense,
        bm25Score: sparse,
        companionBonus: 0.0,
        inertiaBonus: inertia,
        verbKind,
      });
    }

    // 4. Optional code topology boosting
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

    // Preliminary sort descending
    scoredList.sort((a, b) => b.score - a.score);

    // 5. Companion Tool Graph Boosting
    let primaryTool: ImpulseTool | undefined = undefined;
    const companionTools: ImpulseTool[] = [];

    if (scoredList.length > 0 && scoredList[0].score >= opts.minScoreThreshold) {
      const anchor = scoredList[0];
      primaryTool = anchor.tool;
      const anchorScore = anchor.score;

      const neighbors = this.bank.getNeighbors(anchor.tool.name);
      if (neighbors.size > 0) {
        for (const candidate of scoredList) {
          if (candidate.tool.name === anchor.tool.name) continue;

          const edgeWeight = neighbors.get(candidate.tool.name);
          if (edgeWeight && edgeWeight > 0) {
            const bonus = opts.companionBoost * edgeWeight;
            candidate.companionBonus = bonus;
            candidate.score += bonus;

            // Cap companion score so it never overtakes the primary anchor
            if (candidate.score >= anchorScore) {
              candidate.score = anchorScore - 0.001;
            }
            companionTools.push(candidate.tool);
          }
        }
        scoredList.sort((a, b) => b.score - a.score);
      }
    }

    // 6. Family Diversity Filter (Prevent one provider from dominating context)
    const selected: ImpulseTool[] = [];
    const familyCounts = new Map<string, number>();

    // Pass 1: Select top tools respecting family caps and query safety
    for (const candidate of scoredList) {
      if (selected.length >= opts.topK) break;
      if (candidate.score < opts.minScoreThreshold) continue;

      const family = candidate.tool.family || 'default';
      const count = familyCounts.get(family) || 0;

      if (count >= opts.maxPerFamily) {
        continue;
      }

      // Safety check: if user asked a read query, deprioritize destructive deletes
      if (queryIntent === 'read' && candidate.verbKind === 'delete' && candidate.score < 0.85) {
        continue;
      }

      selected.push(candidate.tool);
      familyCounts.set(family, count + 1);
    }

    // Pass 2: Backfill if diversity constraints left slots open
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
