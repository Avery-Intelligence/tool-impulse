import { ToolImpulseEngine } from '../src/core/engine.js';
import { ImpulseTool } from '../src/core/types.js';
import { LocalHashingEmbedder } from '../src/embedders/local-bm25.js';

interface TestCase {
  name: string;
  query: string;
  session?: { priorQuery?: string; recentTools?: string[] };
  expectedTool: string;
  domain: string;
}

const REGISTRY: ImpulseTool[] = [
  // Stripe
  { name: 'stripe_list_invoices', description: 'List customer billing invoices, charges, and payment receipts', family: 'stripe', keywords: ['invoice', 'billing'] },
  { name: 'stripe_get_customer', description: 'Retrieve customer account profile and credit card payment methods', family: 'stripe', keywords: ['customer', 'profile'] },
  { name: 'stripe_refund_charge', description: 'Refund a settled credit card charge to the customer', family: 'stripe', keywords: ['refund', 'reversal'] },
  { name: 'stripe_charge_customer', description: 'Create a new credit card payment charge for customer', family: 'stripe', keywords: ['charge', 'payment'] },

  // Jira
  { name: 'jira_search_issues', description: 'Search Jira tickets, bugs, and backlog tasks using JQL', family: 'jira', keywords: ['ticket', 'bug', 'jql'] },
  { name: 'jira_get_issue', description: 'Get details, comments, and status of a specific Jira issue key', family: 'jira', keywords: ['issue', 'ticket'] },
  { name: 'jira_update_issue', description: 'Modify Jira ticket assignee, priority, status, or description', family: 'jira', keywords: ['update', 'assign'] },
  { name: 'jira_create_issue', description: 'Create a new bug report or task in Jira backlog', family: 'jira', keywords: ['create', 'new'] },

  // GitHub
  { name: 'github_list_prs', description: 'List open and closed pull requests for a repository', family: 'github', keywords: ['pr', 'pull', 'review'] },
  { name: 'github_get_diff', description: 'Inspect code changes and unified git diff patch for a commit or branch', family: 'github', keywords: ['diff', 'patch', 'changes'] },
  { name: 'github_merge_pr', description: 'Merge an approved pull request into the target branch', family: 'github', keywords: ['merge', 'pr'] },

  // Slack
  { name: 'slack_send_message', description: 'Send a formatted message to a Slack channel or direct message', family: 'slack', keywords: ['slack', 'message', 'chat'] },
  { name: 'slack_read_channel', description: 'Fetch conversation history and recent chats from a Slack channel', family: 'slack', keywords: ['slack', 'history', 'read'] },

  // Google Calendar
  { name: 'calendar_list_events', description: 'List upcoming schedule meetings and appointments on Google Calendar', family: 'calendar', keywords: ['calendar', 'meeting', 'schedule'] },
  { name: 'calendar_create_event', description: 'Schedule a new calendar meeting with attendees and invite links', family: 'calendar', keywords: ['calendar', 'invite', 'meeting'] },

  // SQL
  { name: 'sql_read_query', description: 'Execute read-only SQL SELECT queries against PostgreSQL database', family: 'sql', keywords: ['sql', 'query', 'select'] },
  { name: 'sql_list_tables', description: 'Inspect schema structure and list table names in database', family: 'sql', keywords: ['sql', 'schema', 'tables'] },

  // Docker
  { name: 'docker_list_containers', description: 'List running Docker containers, ports, and health statuses', family: 'docker', keywords: ['docker', 'ps', 'container'] },
  { name: 'docker_restart', description: 'Restart an unhealthy container service', family: 'docker', keywords: ['docker', 'restart'] },
];

const TEST_CASES: TestCase[] = [
  // 1. Direct Queries
  { name: 'Stripe Direct', query: 'Show me unpaid customer invoices', expectedTool: 'stripe_list_invoices', domain: 'stripe' },
  { name: 'Jira Direct', query: 'Search open bugs in the backlog', expectedTool: 'jira_search_issues', domain: 'jira' },
  { name: 'GitHub Direct', query: 'List pull requests waiting for review', expectedTool: 'github_list_prs', domain: 'github' },
  { name: 'Slack Direct', query: 'Post incident alert to war-room chat', expectedTool: 'slack_send_message', domain: 'slack' },
  { name: 'Calendar Direct', query: 'What meetings do I have tomorrow?', expectedTool: 'calendar_list_events', domain: 'calendar' },
  { name: 'SQL Direct', query: 'Run SELECT count(*) FROM users', expectedTool: 'sql_read_query', domain: 'sql' },
  { name: 'Docker Direct', query: 'List running containers and port mappings', expectedTool: 'docker_list_containers', domain: 'docker' },

  // 2. Synonyms & Conceptual Matches
  { name: 'Stripe Conceptual', query: 'Reverse the credit card transaction for user 901', expectedTool: 'stripe_refund_charge', domain: 'stripe' },
  { name: 'Calendar Conceptual', query: 'Book a sync with Sarah on Thursday', expectedTool: 'calendar_create_event', domain: 'calendar' },
  { name: 'GitHub Conceptual', query: 'Inspect code changes on branch before landing', expectedTool: 'github_get_diff', domain: 'github' },

  // 3. Multi-Turn Pronoun Shifts
  {
    name: 'Pronoun Shift (Stripe)',
    query: 'Now bill them for the overdue balance',
    session: { priorQuery: 'Look up customer account Acme Corp in Stripe', recentTools: ['stripe_get_customer'] },
    expectedTool: 'stripe_charge_customer',
    domain: 'stripe',
  },
  {
    name: 'Pronoun Shift (Jira)',
    query: 'Assign it to Dave and change priority to high',
    session: { priorQuery: 'Get ticket details for JIRA-551', recentTools: ['jira_get_issue'] },
    expectedTool: 'jira_update_issue',
    domain: 'jira',
  },
];

async function runBenchmarks() {
  console.log('------------------------------------------------------------');
  console.log(' TOOL IMPULSE: REAL-WORLD BENCHMARK SUITE');
  console.log(' Testing in-memory hybrid retrieval, pronoun shifts & chains');
  console.log('------------------------------------------------------------\n');

  const embedder = new LocalHashingEmbedder(256);
  const engine = new ToolImpulseEngine({ embedder });
  await engine.registerTools(REGISTRY);

  // Add companion edges
  engine.addEdges([
    { fromTool: 'jira_get_issue', toTool: 'jira_update_issue', weight: 0.85 },
    { fromTool: 'stripe_get_customer', toTool: 'stripe_charge_customer', weight: 0.80 },
  ]);

  let passed = 0;
  const latencies: number[] = [];

  for (const tc of TEST_CASES) {
    let session;
    if (tc.session) {
      const priorEmb = tc.session.priorQuery ? await embedder.embedQuery(tc.session.priorQuery) : undefined;
      session = {
        priorTurnEmbedding: priorEmb,
        recentToolNames: tc.session.recentTools,
      };
    }

    const start = performance.now();
    const result = await engine.resolve(tc.query, session, { topK: 3 });
    const elapsed = performance.now() - start;
    latencies.push(elapsed);

    const hit = result.tools.some((t) => t.name === tc.expectedTool);
    if (hit) passed++;

    const status = hit ? '✓ PASS' : '✗ FAIL';
    console.log(`[${status}] ${tc.name.padEnd(25)} -> Top: ${result.tools.map((t) => t.name).join(', ')} (${elapsed.toFixed(3)}ms)`);
  }

  latencies.sort((a, b) => a - b);
  const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const p50 = latencies[Math.floor(latencies.length * 0.5)];
  const p95 = latencies[Math.floor(latencies.length * 0.95)];

  console.log('\n------------------------------------------------------------');
  console.log(`Accuracy:  ${passed}/${TEST_CASES.length} (${((passed / TEST_CASES.length) * 100).toFixed(1)}%)`);
  console.log(`Latency:   avg = ${avg.toFixed(3)}ms | p50 = ${p50.toFixed(3)}ms | p95 = ${p95.toFixed(3)}ms`);
  console.log('------------------------------------------------------------\n');

  if (passed < TEST_CASES.length) {
    console.error(`Benchmark failed: expected 100% on core scenarios.`);
    process.exit(1);
  }
}

runBenchmarks().catch((err) => {
  console.error(err);
  process.exit(1);
});
