/**
 * Tool Impulse Evaluation Suite
 *
 * Demonstrates real-world evaluation of tool routing across:
 * 1. Direct domain queries
 * 2. Multi-turn pronoun shifts (trajectory blending)
 * 3. Companion workflow tool chains
 * 4. Domain diversity capping
 * 5. Debug explanation trace
 */

import { ToolImpulse } from '../src/core/engine.js';
import { ToolDefinition } from '../src/core/types.js';

const TOOLS: ToolDefinition[] = [
  // Stripe
  { name: 'stripe_list_invoices', description: 'List customer billing invoices, charges, and payment receipts', domain: 'stripe', keywords: ['invoice', 'billing'] },
  { name: 'stripe_get_customer', description: 'Retrieve customer account profile and credit card payment methods', domain: 'stripe', keywords: ['customer', 'profile'] },
  { name: 'stripe_refund_charge', description: 'Refund a settled credit card charge to the customer', domain: 'stripe', keywords: ['refund', 'reversal'] },
  { name: 'stripe_charge_customer', description: 'Create a new credit card payment charge for customer', domain: 'stripe', keywords: ['charge', 'payment'] },

  // Jira
  { name: 'jira_search_issues', description: 'Search Jira tickets, bugs, and backlog tasks using JQL', domain: 'jira', keywords: ['ticket', 'bug', 'jql'] },
  { name: 'jira_get_issue', description: 'Get details, comments, and status of a specific Jira issue key', domain: 'jira', keywords: ['issue', 'ticket'] },
  { name: 'jira_update_issue', description: 'Modify Jira ticket assignee, priority, status, or description', domain: 'jira', keywords: ['update', 'assign'] },
  { name: 'jira_create_issue', description: 'Create a new bug report or task in Jira backlog', domain: 'jira', keywords: ['create', 'new'] },

  // GitHub
  { name: 'github_list_prs', description: 'List open and closed pull requests for a repository', domain: 'github', keywords: ['pr', 'pull', 'review'] },
  { name: 'github_get_diff', description: 'Inspect unified git diff patch and code changes for a commit or branch', domain: 'github', keywords: ['diff', 'patch', 'changes'] },
  { name: 'github_merge_pr', description: 'Merge an approved pull request into the target branch', domain: 'github', keywords: ['merge', 'pr'] },

  // Slack
  { name: 'slack_send_message', description: 'Send a formatted message to a Slack channel or direct message', domain: 'slack', keywords: ['slack', 'message', 'chat'] },
  { name: 'slack_read_channel', description: 'Fetch conversation history and recent chats from a Slack channel', domain: 'slack', keywords: ['slack', 'history', 'read'] },

  // Database
  { name: 'db_read_query', description: 'Execute read-only SQL SELECT queries against database', domain: 'database', keywords: ['sql', 'query', 'select'] },
  { name: 'db_list_tables', description: 'Inspect schema structure and list table names in database', domain: 'database', keywords: ['sql', 'schema', 'tables'] },
];

async function runEvals() {
  console.log('====================================================');
  console.log(' TOOL IMPULSE: EVALUATION SUITE');
  console.log(' In-memory tool retrieval evaluation across 15 tools');
  console.log('====================================================\n');

  const engine = new ToolImpulse({ tools: TOOLS });

  // Add workflow companion edges
  engine.addWorkflowEdges([
    { fromTool: 'jira_get_issue', toTool: 'jira_update_issue', weight: 0.85 },
    { fromTool: 'stripe_get_customer', toTool: 'stripe_charge_customer', weight: 0.80 },
  ]);

  // 1. Direct match with debug explanation
  console.log('--- 1. Direct Query + Debug Trace ---');
  const r1 = await engine.resolve('Check unpaid customer invoices in Stripe', undefined, { debug: true });
  console.log(r1.explanation);
  console.log(`Mounted tools: [${r1.selectedNames.join(', ')}]\n`);

  // 2. Pronoun shift across multi-turn session
  console.log('--- 2. Pronoun Shift (Trajectory Context) ---');
  console.log('Turn 1: "Get details on customer Acme Corp in Stripe"');
  const t1 = await engine.resolve('Get details on customer Acme Corp in Stripe');
  console.log(`Mounted: [${t1.selectedNames.join(', ')}]`);

  console.log('Turn 2: "Now charge them for the outstanding balance" (contains pronoun "them")');
  const t2 = await engine.resolve('Now charge them for the outstanding balance', {
    recentToolNames: t1.selectedNames,
  }, { debug: true });
  console.log(t2.explanation);
  console.log(`Mounted: [${t2.selectedNames.join(', ')}]\n`);

  // 3. Workflow companion chain
  console.log('--- 3. Companion Workflow Prime ---');
  console.log('Query: "Fetch ticket JIRA-401" -> should prime jira_update_issue');
  const t3 = await engine.resolve('Fetch ticket JIRA-401', undefined, { debug: true });
  console.log(t3.explanation);
  console.log(`Mounted: [${t3.selectedNames.join(', ')}]\n`);

  // 4. Cold start fallback
  console.log('--- 4. Cold Start / Greeting ---');
  console.log('Query: "Hello there, good morning!"');
  const t4 = await engine.resolve('Hello there, good morning!', undefined, {
    defaultTools: ['slack_send_message', 'jira_search_issues'],
    debug: true,
  });
  console.log(t4.explanation);
  console.log(`Mounted: [${t4.selectedNames.join(', ')}]\n`);

  console.log('All evaluations completed successfully.');
}

runEvals().catch(console.error);
