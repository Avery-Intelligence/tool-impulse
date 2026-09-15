import { ToolCatalog } from './bank.js';
import { OkapiBM25 } from './bm25.js';
import {
  RouterOptions,
  ScoredToolMatch,
  SessionState,
  ToolDefinition,
  ToolRouteResult,
} from './types.js';

export class ToolResolver {
  private catalog: ToolCatalog;
  private bm25: OkapiBM25;
  private defaultOptions: Required<Omit<RouterOptions, 'defaultTools' | 'debug' | 'maxPerDomain' | 'domainResolver'>> = {
    topK: 3,
    alpha: 0.70,
    beta: 0.75,
    driftThreshold: 0.35,
    inertiaBonus: 0.20,
    companionBoost: 0.25,
    minScoreThreshold: 0.05,
  };

  constructor(catalog: ToolCatalog) {
    this.catalog = catalog;
    this.bm25 = new OkapiBM25();
  }

  /**
   * Synchronize the Okapi BM25 index with current catalog tools.
   */
  public syncIndex(): void {
    const tools = this.catalog.getAllTools();
    const docs = tools.map((t) => ({
      id: t.name,
      text: `${t.name} ${t.description} ${(t.keywords || []).join(' ')}`,
    }));
    this.bm25.indexDocuments(docs);
  }

  /**
   * Blend prior turn embedding with current query vector to resolve pronoun shifts.
   * Performs cosine similarity drift detection: if the cosine similarity between
   * current and prior query vectors is below driftThreshold (default: 0.35), a topic
   * shift / intent change is detected, and the prior trajectory is safely discarded.
   */
  public blendTrajectory(
    currentEmbedding: Float32Array,
    priorEmbedding?: Float32Array | number[],
    beta: number = 0.75,
    driftThreshold: number = 0.35
  ): Float32Array {
    if (!priorEmbedding) return currentEmbedding;

    const prior = priorEmbedding instanceof Float32Array ? priorEmbedding : new Float32Array(priorEmbedding);
    if (prior.length !== currentEmbedding.length) return currentEmbedding;

    // Cosine similarity check (vectors are L2-normalized)
    let cosSim = 0;
    for (let i = 0; i < currentEmbedding.length; i++) {
      cosSim += currentEmbedding[i] * prior[i];
    }

    // Intent Shift / Topic Switch: Queries are orthogonal or divergent.
    // Discard prior trajectory to avoid cross-domain contamination.
    if (cosSim < driftThreshold) {
      return currentEmbedding;
    }

    const blended = new Float32Array(currentEmbedding.length);
    let sumSq = 0;

    for (let i = 0; i < blended.length; i++) {
      const val = beta * currentEmbedding[i] + (1 - beta) * prior[i];
      blended[i] = val;
      sumSq += val * val;
    }

    const norm = Math.sqrt(sumSq);
    if (norm <= 1e-12 || !isFinite(norm)) {
      return currentEmbedding;
    }
    for (let i = 0; i < blended.length; i++) {
      blended[i] /= norm;
    }

    return blended;
  }

  /**
   * Resolve and rank active tools for an incoming user query.
   */
  public resolve(
    query: string,
    queryEmbedding?: Float32Array,
    session?: SessionState,
    options?: RouterOptions
  ): ToolRouteResult {
    const startTime = performance.now();
    const opts = { ...this.defaultOptions, ...options };

    // 1. Blend trajectory if multi-turn history exists
    let activeVec: Float32Array | undefined = queryEmbedding;
    if (queryEmbedding && session?.priorTurnEmbedding) {
      activeVec = this.blendTrajectory(queryEmbedding, session.priorTurnEmbedding, opts.beta, opts.driftThreshold);
    }

    // 2. Score via Okapi BM25 lexical index
    const bm25Scores = this.bm25.score(query);
    const recentSet = new Set(session?.recentToolNames || []);
    const candidateList: ScoredToolMatch[] = [];
    const allTools = this.catalog.getAllTools();

    // 3. Compute base scores (dense + BM25 + inertia)
    for (const tool of allTools) {
      const dense = activeVec ? this.catalog.computeCosine(activeVec, tool.name) : 0.0;
      const sparse = bm25Scores.get(tool.name) || 0.0;
      const domain = tool.domain || 'default';
      const reasons: string[] = [];

      // Weight dense and sparse matches
      const baseScore = activeVec ? opts.alpha * dense + (1 - opts.alpha) * sparse : sparse;
      if (dense > 0.1) reasons.push(`Dense match (${dense.toFixed(2)})`);
      if (sparse > 0.1) reasons.push(`BM25 match (${sparse.toFixed(2)})`);

      // Inertia bonus for recently executed tools
      const inertia = recentSet.has(tool.name) ? opts.inertiaBonus : 0.0;
      if (inertia > 0) reasons.push(`Inertia bonus (+${inertia.toFixed(2)})`);

      candidateList.push({
        tool,
        totalScore: baseScore + inertia,
        denseScore: dense,
        bm25Score: sparse,
        companionBonus: 0.0,
        inertiaBonus: inertia,
        domain,
        reasons,
      });
    }

    // Sort by preliminary score descending
    candidateList.sort((a, b) => b.totalScore - a.totalScore);

    // 4. Companion Workflow Boosting
    let primaryTool: ToolDefinition | undefined = undefined;

    if (candidateList.length > 0 && candidateList[0].totalScore >= opts.minScoreThreshold) {
      const anchor = candidateList[0];
      primaryTool = anchor.tool;
      const anchorScore = anchor.totalScore;

      const neighbors = this.catalog.getNeighbors(anchor.tool.name);
      if (neighbors.size > 0) {
        for (const candidate of candidateList) {
          if (candidate.tool.name === anchor.tool.name) continue;

          const edgeWeight = neighbors.get(candidate.tool.name);
          if (edgeWeight && edgeWeight > 0) {
            const bonus = opts.companionBoost * edgeWeight;
            candidate.companionBonus = bonus;
            candidate.totalScore += bonus;
            candidate.reasons.push(`Companion boost from ${anchor.tool.name} (+${bonus.toFixed(2)})`);

            // Anchor ceiling: companion tool score is capped so it never outranks the direct primary anchor
            const maxCompanionScore = anchorScore * 0.95;
            if (candidate.totalScore > maxCompanionScore) {
              candidate.totalScore = maxCompanionScore;
            }
          }
        }
        candidateList.sort((a, b) => b.totalScore - a.totalScore);
      }
    }

    // 5. Domain Selection & Optional Capping
    const selected: ToolDefinition[] = [];
    const domainCounts = new Map<string, number>();
    const maxAllowedPerDomain = opts.maxPerDomain;

    for (const candidate of candidateList) {
      if (selected.length >= opts.topK) break;
      if (candidate.totalScore < opts.minScoreThreshold) continue;

      const domain = opts.domainResolver ? opts.domainResolver(candidate.tool) : candidate.domain;

      if (maxAllowedPerDomain !== undefined && domain) {
        const count = domainCounts.get(domain) || 0;
        if (count >= maxAllowedPerDomain) {
          continue;
        }
        domainCounts.set(domain, count + 1);
      }

      selected.push(candidate.tool);
    }

    // 6. Cold Start / Zero Match Handling
    if (selected.length === 0 && options?.defaultTools && options.defaultTools.length > 0) {
      for (const name of options.defaultTools) {
        if (selected.length >= opts.topK) break;

        const fallback = this.catalog.getTool(name);
        if (fallback) {
          const domain = opts.domainResolver ? opts.domainResolver(fallback) : fallback.domain;
          if (maxAllowedPerDomain !== undefined && domain) {
            const count = domainCounts.get(domain) || 0;
            if (count >= maxAllowedPerDomain) {
              continue;
            }
            domainCounts.set(domain, count + 1);
          }
          selected.push(fallback);
        }
      }
    }

    const latencyMs = performance.now() - startTime;
    const scores: Record<string, number> = {};
    for (const c of candidateList) {
      scores[c.tool.name] = Number(c.totalScore.toFixed(3));
    }

    // 7. Optional Human-Readable Debug Trace
    let explanation: string | undefined = undefined;
    if (options?.debug) {
      const lines = [`[ToolImpulse Debug] Query: "${query}" (resolved in ${latencyMs.toFixed(3)}ms)`];
      if (selected.length === 0) {
        lines.push('  No tools matched threshold. Cold start fallback applied.');
      } else {
        selected.forEach((tool, idx) => {
          const match = candidateList.find((c) => c.tool.name === tool.name);
          lines.push(`  ${idx + 1}. ${tool.name} (Score: ${match?.totalScore.toFixed(3)})`);
          if (match?.reasons.length) {
            match.reasons.forEach((r) => lines.push(`     - ${r}`));
          }
        });
      }
      explanation = lines.join('\n');
    }

    return {
      tools: selected,
      selectedNames: selected.map((t) => t.name),
      primaryTool,
      queryEmbedding: activeVec,
      scores,
      explanation,
      latencyMs,
    };
  }
}
