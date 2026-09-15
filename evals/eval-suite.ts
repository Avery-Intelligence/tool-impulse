/**
 * Tool Impulse Real-World Scenario Evaluation
 *
 * Evaluates tool routing accuracy, multi-turn tracking, and domain filtering
 * against realistic tool catalogs and queries adapted from the
 * Berkeley Function Calling Leaderboard (BFCL).
 */

import { ToolImpulse } from '../src/core/engine.js';
import { BFCL_TOOLS, BFCL_BENCHMARK_CASES } from './datasets/bfcl-tools.js';

async function runEvaluations() {
  console.log('================================================================');
  console.log(' TOOL IMPULSE: EVALUATION SUITE');
  console.log(` Dataset: ${BFCL_TOOLS.length} real-world tools across 9 functional domains`);
  console.log(` Benchmark: ${BFCL_BENCHMARK_CASES.length} test queries (exact, synonym, conversational)`);
  console.log('================================================================\n');

  const engine = new ToolImpulse({ tools: BFCL_TOOLS });

  let totalQueries = 0;
  let top1Hits = 0;
  let top3Hits = 0;
  const latencies: number[] = [];

  console.log('1. Berkeley Function Calling Leaderboard (BFCL) Retrieval Accuracy:');
  console.log('----------------------------------------------------------------');

  for (const testCase of BFCL_BENCHMARK_CASES) {
    totalQueries++;
    const t0 = performance.now();
    const result = engine.resolveSync(testCase.query, undefined, undefined, { topK: 3 });
    const latency = performance.now() - t0;
    latencies.push(latency);

    const isTop1 =
      result.selectedNames[0] === testCase.expectedTopTool ||
      (testCase.acceptableTools && testCase.acceptableTools.includes(result.selectedNames[0]));

    const isTop3 =
      result.selectedNames.includes(testCase.expectedTopTool) ||
      (testCase.acceptableTools && testCase.acceptableTools.some((t) => result.selectedNames.includes(t)));

    if (isTop1) top1Hits++;
    if (isTop3) top3Hits++;

    const statusMark = isTop1 ? '✓ [Top-1]' : isTop3 ? '✓ [Top-3]' : '✗ [MISS]';
    console.log(`  ${statusMark} "${testCase.query}"`);
    console.log(`     → Selected: [${result.selectedNames.join(', ')}] (Expected: ${testCase.expectedTopTool})`);
  }

  const top1Accuracy = (top1Hits / totalQueries) * 100;
  const top3Accuracy = (top3Hits / totalQueries) * 100;
  latencies.sort((a, b) => a - b);
  const p50Latency = latencies[Math.floor(latencies.length * 0.5)];
  const p95Latency = latencies[Math.floor(latencies.length * 0.95)];

  console.log('\n  Benchmark Summary:');
  console.log(`    Total queries evaluated: ${totalQueries}`);
  console.log(`    Top-1 Accuracy:          ${top1Accuracy.toFixed(1)}% (${top1Hits}/${totalQueries})`);
  console.log(`    Top-3 Accuracy:          ${top3Accuracy.toFixed(1)}% (${top3Hits}/${totalQueries})`);
  console.log(`    Latency P50:             ${p50Latency.toFixed(3)}ms`);
  console.log(`    Latency P95:             ${p95Latency.toFixed(3)}ms\n`);

  // --------------------------------------------------------------------------
  // 2. Multi-Turn Follow-Ups with Orthogonal Vocabulary
  // --------------------------------------------------------------------------
  console.log('2. Multi-Turn Follow-Ups & Orthogonal Vocabulary:');
  console.log('----------------------------------------------------------------');

  // Turn 1: Database query
  const turn1Query = 'Query user signups from the accounts table in Postgres';
  const t1 = engine.resolveSync(turn1Query, undefined, undefined, { topK: 2 });
  console.log(`  Turn 1 Query: "${turn1Query}"`);
  console.log(`  Turn 1 Selected: [${t1.selectedNames.join(', ')}]`);

  // Turn 2: Follow-up command with completely orthogonal vocabulary ("email that report to Alice")
  const turn2Query = 'now email that report to Alice';
  const t2 = engine.resolveSync(
    turn2Query,
    undefined,
    { recentToolNames: t1.selectedNames }, // tool inertia
    { topK: 3 }
  );
  console.log(`  Turn 2 Query: "${turn2Query}"`);
  console.log(`  Turn 2 Selected (with inertia): [${t2.selectedNames.join(', ')}]`);

  const hasNewAction = t2.selectedNames.includes('slack_send_message');
  const hasPriorInertia = t2.selectedNames.includes('sql_execute_query');

  if (hasNewAction && hasPriorInertia) {
    console.log('  ✓ Preserved prior database tool via inertia while mounting new messaging tool\n');
  } else {
    console.log(`  - Multi-turn result: [${t2.selectedNames.join(', ')}]\n`);
  }

  // --------------------------------------------------------------------------
  // 3. Companion Graph Activation (Opt-In)
  // --------------------------------------------------------------------------
  console.log('3. Companion Tool Priming (Opt-In):');
  console.log('----------------------------------------------------------------');

  engine.addWorkflowEdges([
    { fromTool: 'jira_get_issue', toTool: 'jira_transition_issue', weight: 0.9 },
  ]);

  // Default: companionBoost is 0.0 -> no artificial companion boost
  const rDefault = engine.resolveSync('Inspect ticket details for PROD-102', undefined, undefined, { topK: 3 });
  console.log(`  Default (companionBoost: 0.0): [${rDefault.selectedNames.join(', ')}]`);
  const companionNotBoostedByDefault = rDefault.scores['jira_transition_issue'] === 0 || !rDefault.selectedNames.includes('jira_transition_issue');
  console.log(`  ✓ Default mode avoids artificial companion boosting: ${companionNotBoostedByDefault}`);

  // Opt-in: companionBoost explicitly enabled
  const rBoosted = engine.resolveSync('Inspect ticket details for PROD-102', undefined, undefined, {
    topK: 3,
    companionBoost: 0.30,
  });
  console.log(`  Opt-in (companionBoost: 0.30): [${rBoosted.selectedNames.join(', ')}]`);
  const companionBoosted = rBoosted.selectedNames.includes('jira_transition_issue');
  console.log(`  ✓ Explicit companionBoost mounts downstream transition tool: ${companionBoosted}\n`);

  // --------------------------------------------------------------------------
  // 4. Domain Diversity Capping
  // --------------------------------------------------------------------------
  console.log('4. Domain Diversity Capping (maxPerDomain):');
  console.log('----------------------------------------------------------------');
  const crossDomainQuery = 'Check unpaid invoices in Stripe and notify the customer in Slack';
  const cappedResult = engine.resolveSync(crossDomainQuery, undefined, undefined, {
    topK: 4,
    maxPerDomain: 1,
  });

  const domainsFound = cappedResult.tools.map((t) => t.domain);
  const uniqueDomains = new Set(domainsFound);
  const respectedCap = domainsFound.length === uniqueDomains.size;

  console.log(`  Query: "${crossDomainQuery}"`);
  console.log(`  Selected (maxPerDomain: 1): [${cappedResult.selectedNames.join(', ')}]`);
  console.log(`  Domains: [${domainsFound.join(', ')}]`);
  console.log(`  ✓ Domain diversity cap respected (1 tool per domain): ${respectedCap}\n`);

  console.log('================================================================');
  console.log(' EVALUATION COMPLETE');
  console.log('================================================================');
}

runEvaluations().catch(console.error);
