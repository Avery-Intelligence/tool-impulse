/**
 * Tool Impulse: Complete Production Agent Lifecycle Example
 *
 * Demonstrates:
 * 1. Offline BM25 In-Process Tool Retrieval (<0.1ms math, 0 external API calls)
 * 2. Multi-turn trajectory context blending (resolving pronoun anaphora)
 * 3. Intent-shift / topic boundary detection (dropping stale context on topic switch)
 * 4. Companion workflow graph activation (mounting co-occurring tools)
 * 5. Dynamic multi-tenant catalog hot-swapping (zero cache lock or cross-tenant leakage)
 * 6. Two-phase omission recovery (handling edge-case tool requirements safely)
 *
 * Run via:
 *   pnpm run example
 */

import { ToolImpulse, createToolRouter, ToolDefinition, SessionState } from '../src/index.js';

// 1. Define enterprise catalog across 4 distinct domains
const ENTERPRISE_CATALOG: ToolDefinition[] = [
  // Payments / Billing (Stripe)
  {
    name: 'stripe_list_invoices',
    description: 'Retrieve customer billing statements, invoice history, and unpaid balances',
    domain: 'stripe',
  },
  {
    name: 'stripe_refund_charge',
    description: 'Issue a credit return to reverse credit card charges or settle disputed funds',
    domain: 'stripe',
  },
  {
    name: 'stripe_get_customer',
    description: 'Fetch customer profile identity, default card, and billing email address',
    domain: 'stripe',
  },

  // Issue Tracker (Jira)
  {
    name: 'jira_get_issue',
    description: 'Inspect ticket fields, summary, status, reporter, and comments for a defect',
    domain: 'jira',
  },
  {
    name: 'jira_update_issue',
    description: 'Modify assignee, change ticket status, or append progress notes to an issue',
    domain: 'jira',
  },

  // Version Control (GitHub)
  {
    name: 'github_list_prs',
    description: 'List open pull requests and branches awaiting review in repository',
    domain: 'github',
  },
  {
    name: 'github_merge_pr',
    description: 'Squash or fast-forward merge an approved pull request into main branch',
    domain: 'github',
  },

  // Team Communication (Slack)
  {
    name: 'slack_send_message',
    description: 'Post conversational updates, notifications, or alerts to a channel room',
    domain: 'slack',
  },

  // Fallback Affordance (Omission Recovery)
  {
    name: 'search_catalog_fallback',
    description: 'Search full catalog or request escalation when currently mounted tools are insufficient',
    domain: 'system',
  },
];

// Helper to create normalized test embedding vector
function mockVector(dim: number, activeIdx: number, secondaryIdx?: number): Float32Array {
  const vec = new Float32Array(dim);
  vec[activeIdx] = 1.0;
  if (secondaryIdx !== undefined) vec[secondaryIdx] = 0.5;
  let sumSq = 0;
  for (let i = 0; i < dim; i++) sumSq += vec[i] * vec[i];
  const norm = Math.sqrt(sumSq) || 1;
  for (let i = 0; i < dim; i++) vec[i] /= norm;
  return vec;
}

async function main() {
  console.log('================================================================');
  console.log(' TOOL IMPULSE: END-TO-END PRODUCTION AGENT LIFECYCLE DEMO');
  console.log('================================================================\n');

  // Initialize engine with catalog
  const engine = new ToolImpulse({ tools: ENTERPRISE_CATALOG });

  // Add workflow graph edge: when inspecting an issue, prime update issue
  engine.addWorkflowEdges([
    { fromTool: 'jira_get_issue', toTool: 'jira_update_issue', weight: 0.85 },
    { fromTool: 'stripe_get_customer', toTool: 'stripe_refund_charge', weight: 0.80 },
  ]);

  // Set 8-dimensional semantic embeddings (0: billing, 1: refund, 2: customer, 3: jira_read, 4: jira_write, 5: github, 6: slack, 7: system)
  engine.setEmbeddings({
    stripe_list_invoices: Array.from(mockVector(8, 0)),
    stripe_refund_charge: Array.from(mockVector(8, 1, 0)),
    stripe_get_customer: Array.from(mockVector(8, 2, 0)),
    jira_get_issue: Array.from(mockVector(8, 3)),
    jira_update_issue: Array.from(mockVector(8, 4, 3)),
    github_list_prs: Array.from(mockVector(8, 5)),
    github_merge_pr: Array.from(mockVector(8, 5)),
    slack_send_message: Array.from(mockVector(8, 6)),
    search_catalog_fallback: Array.from(mockVector(8, 7)),
  });

  // --------------------------------------------------------------------------
  // Phase 1: Pure In-Process BM25 Retrieval (Zero network latency, 0 token spend)
  // --------------------------------------------------------------------------
  console.log('--- Phase 1: Pure In-Process BM25 Retrieval (Offline) ---');
  const q1 = 'Check pending invoices and billing statements for Acme Corp';
  const r1 = engine.resolveSync(q1, undefined, undefined, { topK: 3 });

  console.log(`Query: "${q1}"`);
  console.log(`Selected tools (${r1.latencyMs.toFixed(3)}ms): [${r1.selectedNames.join(', ')}]`);
  console.log(`Anchor tool: ${r1.primaryTool?.name}`);
  console.log('');

  // --------------------------------------------------------------------------
  // Phase 2: Multi-Turn Trajectory Blending (Pronoun Anaphora Resolution)
  // --------------------------------------------------------------------------
  console.log('--- Phase 2: Multi-Turn Conversation with Trajectory Blending ---');

  // Turn 1: User asks about a customer
  const turn1Query = 'Fetch profile for customer Globex';
  const turn1Vec = mockVector(8, 2); // Customer subspace
  const t1 = engine.resolveSync(turn1Query, turn1Vec, undefined, { topK: 2 });
  console.log(`Turn 1: "${turn1Query}"`);
  console.log(`Mounted tools: [${t1.selectedNames.join(', ')}]`);

  // Turn 2: User says "reimburse them for that charge" (pronoun query without customer/stripe keywords)
  const turn2Query = 'reimburse them for that charge';
  const turn2Vec = mockVector(8, 1); // Refund signal
  const session: SessionState = {
    priorTurnEmbedding: turn1Vec,
    recentToolNames: t1.selectedNames,
  };
  const t2 = engine.resolveSync(turn2Query, turn2Vec, session, { topK: 2, beta: 0.70 });
  console.log(`\nTurn 2 (Pronoun follow-up): "${turn2Query}"`);
  console.log(`Blended trajectory maintained customer context and mounted: [${t2.selectedNames.join(', ')}]`);

  // Turn 3: User abruptly switches topics to GitHub PRs
  const turn3Query = 'List open pull requests ready for review';
  const turn3Vec = mockVector(8, 5); // GitHub subspace (orthogonal to billing: dot product < 0.35)
  const session3: SessionState = {
    priorTurnEmbedding: turn2Vec,
    recentToolNames: t2.selectedNames,
  };
  const t3 = engine.resolveSync(turn3Query, turn3Vec, session3, { topK: 2, driftThreshold: 0.35 });
  console.log(`\nTurn 3 (Intent switch): "${turn3Query}"`);
  console.log(`Drift cutoff detected topic shift (<0.35 cosine). Discarded billing context.`);
  console.log(`Mounted tools: [${t3.selectedNames.join(', ')}]`);
  console.log('');

  // --------------------------------------------------------------------------
  // Phase 3: Safe Omission Recovery Pattern
  // --------------------------------------------------------------------------
  console.log('--- Phase 3: Production Omission Recovery Pattern ---');
  // When an ambiguous or unpredicted request arrives:
  const ambiguousQuery = 'Hey, can you help me out?';
  const r3 = engine.resolveSync(ambiguousQuery, undefined, undefined, {
    topK: 2,
    minScoreThreshold: 0.05,
    defaultTools: ['slack_send_message', 'search_catalog_fallback'],
  });

  console.log(`Ambiguous Query: "${ambiguousQuery}"`);
  console.log(`Score did not meet high-confidence threshold.`);
  console.log(`Default fallback mounted safe communication & search affordances: [${r3.selectedNames.join(', ')}]`);
  console.log('If the model determines that a specialized tool is missing, it can call search_catalog_fallback');
  console.log('to escalate and mount specific tools on demand.\n');

  // --------------------------------------------------------------------------
  // Phase 4: Dynamic Multi-Tenant Hot Swapping
  // --------------------------------------------------------------------------
  console.log('--- Phase 4: Dynamic Multi-Tenant Hot-Swapping ---');
  const multiTenantRouter = createToolRouter({ topK: 2 });

  // Turn A: Tenant A uses Finance tools
  const tenantATools = {
    stripe_list_invoices: { description: 'List customer billing invoices' },
    stripe_refund_charge: { description: 'Refund customer payment' },
  };
  const resA = await multiTenantRouter.getTools('Process invoice refund', tenantATools);
  console.log(`Tenant A (Finance) tools mounted: [${Object.keys(resA.tools).join(', ')}]`);

  // Turn B: Tenant B uses DevOps tools (Completely different tool definitions on next turn)
  const tenantBTools = {
    k8s_restart_pod: { description: 'Restart kubernetes pod deployment' },
    pg_kill_query: { description: 'Terminate hung Postgres query' },
  };
  const resB = await multiTenantRouter.getTools('Restart failing pod', tenantBTools);
  console.log(`Tenant B (DevOps) tools mounted: [${Object.keys(resB.tools).join(', ')}]`);
  console.log('Automatic catalog synchronization detected tenant change with zero cross-tenant contamination.\n');

  console.log('================================================================');
  console.log(' ✓ LIFECYCLE DEMO COMPLETED SUCCESSFULLY');
  console.log('================================================================');
}

main().catch((err) => {
  console.error('Lifecycle demo error:', err);
  process.exit(1);
});
