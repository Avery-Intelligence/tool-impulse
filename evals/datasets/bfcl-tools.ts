import { ToolDefinition } from '../../src/core/types.js';

/**
 * Real-world tool schemas and function signatures adapted from the
 * Berkeley Function Calling Leaderboard (BFCL) and standard agent benchmarks.
 *
 * Covers 9 distinct real-world service domains:
 * - File System & OS
 * - Weather & Geo
 * - Payments & Billing (Stripe)
 * - Relational Databases (Postgres/SQL)
 * - Issue Tracking (Jira)
 * - Version Control (GitHub)
 * - Team Messaging (Slack)
 * - Cloud Infrastructure (AWS S3 & Kubernetes)
 * - Currency & Math
 */
export const BFCL_TOOLS: ToolDefinition[] = [
  // --- File System & OS ---
  {
    name: 'fs_read_file',
    description: 'Read contents of a file from local filesystem as UTF-8 text or binary buffer',
    domain: 'fs',
    keywords: ['file', 'read', 'open', 'filesystem', 'contents', 'cat'],
    readOnly: true,
  },
  {
    name: 'fs_write_file',
    description: 'Write or overwrite text or binary data to a file at specified path',
    domain: 'fs',
    keywords: ['file', 'write', 'save', 'create', 'filesystem'],
  },
  {
    name: 'fs_list_directory',
    description: 'List files and subdirectories located within a target folder',
    domain: 'fs',
    keywords: ['dir', 'directory', 'list', 'folder', 'ls', 'filesystem'],
    readOnly: true,
  },
  {
    name: 'fs_delete_file',
    description: 'Remove a file or directory tree permanently from the filesystem',
    domain: 'fs',
    keywords: ['file', 'delete', 'remove', 'unlink', 'rm'],
  },

  // --- Weather & Geo ---
  {
    name: 'get_current_weather',
    description: 'Get current temperature, humidity, wind speed, and weather condition for a city or coordinates',
    domain: 'weather',
    keywords: ['weather', 'temperature', 'forecast', 'climate', 'celsius', 'fahrenheit'],
    readOnly: true,
  },
  {
    name: 'get_weather_forecast',
    description: 'Get 7-day daily weather forecast including high/low temperatures and precipitation chances',
    domain: 'weather',
    keywords: ['forecast', 'weather', 'rain', 'precipitation', 'weekly', 'outlook'],
    readOnly: true,
  },

  // --- Payments & Billing (Stripe) ---
  {
    name: 'stripe_create_charge',
    description: 'Authorize and capture payment charge against a credit card or payment method token',
    domain: 'stripe',
    keywords: ['stripe', 'charge', 'pay', 'credit_card', 'payment', 'transaction'],
  },
  {
    name: 'stripe_refund_payment',
    description: 'Issue a full or partial monetary refund for a previously captured charge ID',
    domain: 'stripe',
    keywords: ['stripe', 'refund', 'reverse', 'reimburse', 'chargeback', 'credit'],
  },
  {
    name: 'stripe_retrieve_customer',
    description: 'Fetch customer account details, stored payment methods, and invoice history by customer ID',
    domain: 'stripe',
    keywords: ['stripe', 'customer', 'account', 'profile', 'billing', 'invoices'],
    readOnly: true,
  },
  {
    name: 'stripe_list_invoices',
    description: 'Query billing invoices filtered by customer, payment status, or date range',
    domain: 'stripe',
    keywords: ['stripe', 'invoice', 'billing', 'statement', 'unpaid', 'due'],
    readOnly: true,
  },

  // --- Database & SQL ---
  {
    name: 'sql_execute_query',
    description: 'Execute a read-only SQL query against PostgreSQL or MySQL database and return rows',
    domain: 'database',
    keywords: ['sql', 'query', 'select', 'postgres', 'database', 'table', 'records'],
    readOnly: true,
  },
  {
    name: 'sql_explain_query',
    description: 'Generate execution plan and cost estimates (EXPLAIN ANALYZE) for a SQL statement',
    domain: 'database',
    keywords: ['sql', 'explain', 'analyze', 'performance', 'query_plan', 'index'],
    readOnly: true,
  },
  {
    name: 'db_list_tables',
    description: 'List all database tables, columns, indexes, and schema definitions',
    domain: 'database',
    keywords: ['database', 'schema', 'tables', 'columns', 'catalog', 'structure'],
    readOnly: true,
  },

  // --- Issue Tracking (Jira) ---
  {
    name: 'jira_create_issue',
    description: 'Create a new bug report, feature ticket, or task in Jira project backlog',
    domain: 'jira',
    keywords: ['jira', 'ticket', 'issue', 'bug', 'task', 'backlog', 'create'],
  },
  {
    name: 'jira_get_issue',
    description: 'Retrieve issue description, status, priority, reporter, and comments by ticket key',
    domain: 'jira',
    keywords: ['jira', 'ticket', 'issue', 'details', 'inspect', 'summary'],
    readOnly: true,
  },
  {
    name: 'jira_transition_issue',
    description: 'Transition Jira ticket workflow state (e.g., to In Progress, In Review, or Closed)',
    domain: 'jira',
    keywords: ['jira', 'ticket', 'status', 'transition', 'workflow', 'close', 'reopen'],
  },
  {
    name: 'jira_add_comment',
    description: 'Append a public or internal comment to an existing Jira ticket discussion thread',
    domain: 'jira',
    keywords: ['jira', 'comment', 'ticket', 'reply', 'note', 'message'],
  },

  // --- Version Control (GitHub) ---
  {
    name: 'github_create_pull_request',
    description: 'Open a new GitHub pull request comparing a feature branch against the base branch',
    domain: 'github',
    keywords: ['github', 'pr', 'pull_request', 'branch', 'merge', 'open'],
  },
  {
    name: 'github_merge_pull_request',
    description: 'Squash or fast-forward merge an approved GitHub pull request into main branch',
    domain: 'github',
    keywords: ['github', 'pr', 'merge', 'pull_request', 'squash', 'ship'],
  },
  {
    name: 'github_get_diff',
    description: 'Fetch unified diff and list of modified files for a pull request or commit SHA',
    domain: 'github',
    keywords: ['github', 'diff', 'patch', 'changes', 'commit', 'hunks'],
    readOnly: true,
  },
  {
    name: 'github_search_code',
    description: 'Search repository code for function definitions, variables, or regex patterns',
    domain: 'github',
    keywords: ['github', 'code', 'search', 'grep', 'symbol', 'repo'],
    readOnly: true,
  },

  // --- Team Messaging (Slack) ---
  {
    name: 'slack_send_message',
    description: 'Broadcast a message, announcement, or alert to a Slack channel or direct message',
    domain: 'slack',
    keywords: ['slack', 'message', 'post', 'chat', 'channel', 'notify', 'alert'],
  },
  {
    name: 'slack_read_channel',
    description: 'Fetch recent conversation message history and replies from a Slack channel',
    domain: 'slack',
    keywords: ['slack', 'channel', 'history', 'read', 'messages', 'chat', 'timeline'],
    readOnly: true,
  },

  // --- Cloud Infrastructure (AWS & Kubernetes) ---
  {
    name: 'aws_s3_upload_object',
    description: 'Upload a file or data stream to an Amazon S3 storage bucket with specified key',
    domain: 'aws',
    keywords: ['aws', 's3', 'bucket', 'upload', 'storage', 'file'],
  },
  {
    name: 'aws_s3_get_object',
    description: 'Download an object data stream and metadata from an Amazon S3 storage bucket',
    domain: 'aws',
    keywords: ['aws', 's3', 'bucket', 'download', 'fetch', 'object'],
    readOnly: true,
  },
  {
    name: 'k8s_get_pod_logs',
    description: 'Stream stdout and stderr container logs from a Kubernetes pod in a namespace',
    domain: 'k8s',
    keywords: ['k8s', 'kubernetes', 'pod', 'logs', 'container', 'stderr', 'stdout'],
    readOnly: true,
  },
  {
    name: 'k8s_restart_pod',
    description: 'Trigger a rolling restart or pod deletion to reload configuration or recover state',
    domain: 'k8s',
    keywords: ['k8s', 'kubernetes', 'pod', 'restart', 'reboot', 'kill', 'deployment'],
  },

  // --- Currency & Conversion ---
  {
    name: 'convert_currency',
    description: 'Calculate monetary exchange rates and convert amount between two fiat or crypto currencies',
    domain: 'finance',
    keywords: ['currency', 'exchange', 'convert', 'rates', 'forex', 'usd', 'eur'],
    readOnly: true,
  },
];

/**
 * Realistic benchmark test cases derived from Berkeley Function Calling Leaderboard (BFCL)
 * queries, evaluating single-turn tool selection, synonym handling, and multi-domain queries.
 */
export interface BfclBenchmarkCase {
  query: string;
  expectedTopTool: string;
  acceptableTools?: string[];
  domain: string;
  scenario: 'exact' | 'synonym' | 'conversational' | 'cross_domain';
}

export const BFCL_BENCHMARK_CASES: BfclBenchmarkCase[] = [
  // Exact & direct intent
  {
    query: 'What is the temperature and weather in Tokyo today?',
    expectedTopTool: 'get_current_weather',
    domain: 'weather',
    scenario: 'exact',
  },
  {
    query: 'Give me the 7-day weekly forecast for Seattle',
    expectedTopTool: 'get_weather_forecast',
    acceptableTools: ['get_current_weather'],
    domain: 'weather',
    scenario: 'exact',
  },
  {
    query: 'Fetch lines from /etc/hosts on the local machine',
    expectedTopTool: 'fs_read_file',
    domain: 'fs',
    scenario: 'exact',
  },
  {
    query: 'List all files and folders in /var/log directory',
    expectedTopTool: 'fs_list_directory',
    domain: 'fs',
    scenario: 'exact',
  },
  {
    query: 'Delete the temporary cache directory /tmp/build-cache',
    expectedTopTool: 'fs_delete_file',
    domain: 'fs',
    scenario: 'exact',
  },

  // Synonym & paraphrasing (testing Porter stemming & semantic routing)
  {
    query: 'Reverse the monetary payment for customer invoice inv_9921',
    expectedTopTool: 'stripe_refund_payment',
    acceptableTools: ['stripe_list_invoices'],
    domain: 'stripe',
    scenario: 'synonym',
  },
  {
    query: 'Look up past billing statements and unpaid charges for customer cus_412',
    expectedTopTool: 'stripe_list_invoices',
    acceptableTools: ['stripe_retrieve_customer'],
    domain: 'stripe',
    scenario: 'synonym',
  },
  {
    query: 'Profile and run EXPLAIN ANALYZE on the slow users query',
    expectedTopTool: 'sql_explain_query',
    acceptableTools: ['sql_execute_query'],
    domain: 'database',
    scenario: 'synonym',
  },
  {
    query: 'Inspect database schema to see all existing tables and column types',
    expectedTopTool: 'db_list_tables',
    acceptableTools: ['sql_execute_query'],
    domain: 'database',
    scenario: 'synonym',
  },
  {
    query: 'Convert 450 Euros to US Dollars with live exchange rates',
    expectedTopTool: 'convert_currency',
    domain: 'finance',
    scenario: 'synonym',
  },

  // Conversational prompts with filler words
  {
    query: 'Hey could you check the container logs for the failing auth pod in k8s?',
    expectedTopTool: 'k8s_get_pod_logs',
    acceptableTools: ['k8s_restart_pod'],
    domain: 'k8s',
    scenario: 'conversational',
  },
  {
    query: 'Please perform a rolling reboot on the payment-service deployment pod',
    expectedTopTool: 'k8s_restart_pod',
    acceptableTools: ['k8s_get_pod_logs'],
    domain: 'k8s',
    scenario: 'conversational',
  },
  {
    query: 'Can you file a critical bug ticket in Jira for the checkout crash?',
    expectedTopTool: 'jira_create_issue',
    domain: 'jira',
    scenario: 'conversational',
  },
  {
    query: 'Move ticket PROD-1049 status to Closed in the workflow',
    expectedTopTool: 'jira_transition_issue',
    acceptableTools: ['jira_get_issue'],
    domain: 'jira',
    scenario: 'conversational',
  },
  {
    query: 'Post an urgent incident alert into the #production-alerts channel',
    expectedTopTool: 'slack_send_message',
    domain: 'slack',
    scenario: 'conversational',
  },
  {
    query: 'Upload the monthly financial backup archive to our S3 storage bucket',
    expectedTopTool: 'aws_s3_upload_object',
    domain: 'aws',
    scenario: 'conversational',
  },
  {
    query: 'Inspect unified diff and changed hunks for pull request #42',
    expectedTopTool: 'github_get_diff',
    acceptableTools: ['github_create_pull_request'],
    domain: 'github',
    scenario: 'conversational',
  },
  {
    query: 'Squash and merge pull request #88 into main branch',
    expectedTopTool: 'github_merge_pull_request',
    acceptableTools: ['github_get_diff'],
    domain: 'github',
    scenario: 'conversational',
  },
];
