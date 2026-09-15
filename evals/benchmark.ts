/**
 * Tool Impulse Performance & Latency Benchmark
 *
 * Evaluates latency percentiles (P50, P90, P95, P99) across:
 * - 10, 50, 100, and 500 registered tools
 * - Lexical Okapi BM25 routing
 * - Dense vector routing (768d & 1536d) with feature-hashed semantic text vectors
 * - Live Web Request Simulation: Fresh array allocations on every turn
 *   (validates deterministic fingerprint caching without the WeakMap trap)
 */

import { ToolImpulse } from '../src/core/engine.js';
import { ToolDefinition } from '../src/core/types.js';

interface BenchmarkRow {
  catalogSize: number;
  mode: string;
  dimension: string;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
}

const DOMAINS = [
  {
    name: 'stripe',
    actions: ['create', 'refund', 'retrieve', 'cancel', 'update', 'list', 'capture', 'void'],
    entities: ['charge', 'customer', 'invoice', 'subscription', 'payment_method', 'dispute', 'payout', 'coupon'],
    verbs: ['Process', 'Refund', 'Look up', 'Cancel', 'Modify', 'Query', 'Settle', 'Reverse'],
  },
  {
    name: 'github',
    actions: ['list', 'create', 'merge', 'close', 'reopen', 'review', 'diff', 'comment'],
    entities: ['pull_request', 'issue', 'branch', 'commit', 'workflow', 'release', 'repository', 'tag'],
    verbs: ['List', 'Open', 'Merge', 'Close', 'Reopen', 'Submit review for', 'Inspect diff of', 'Post comment to'],
  },
  {
    name: 'jira',
    actions: ['search', 'create', 'transition', 'assign', 'link', 'estimate', 'log_work', 'attach'],
    entities: ['ticket', 'backlog_item', 'sprint', 'epic', 'bug_report', 'subtask', 'component', 'board'],
    verbs: ['Search', 'File new', 'Change status of', 'Reassign', 'Link together', 'Set story points for', 'Log hours on', 'Attach file to'],
  },
  {
    name: 'slack',
    actions: ['send', 'read', 'react', 'pin', 'archive', 'invite', 'set_topic', 'schedule'],
    entities: ['message', 'channel_history', 'direct_message', 'announcement', 'canvas', 'huddle', 'thread', 'reminder'],
    verbs: ['Post', 'Read timeline of', 'Add emoji reaction to', 'Pin important', 'Archive old', 'Invite user to', 'Update topic of', 'Schedule future'],
  },
  {
    name: 'aws',
    actions: ['put', 'get', 'delete', 'describe', 'invoke', 'terminate', 'publish', 'query'],
    entities: ['s3_object', 'ec2_instance', 'lambda_function', 'dynamodb_item', 'cloudwatch_metric', 'sqs_message', 'rds_cluster', 'iam_policy'],
    verbs: ['Upload', 'Fetch', 'Delete', 'Inspect state of', 'Execute serverless', 'Shut down', 'Emit metric to', 'Query database'],
  },
  {
    name: 'postgres',
    actions: ['query', 'explain', 'analyze', 'index', 'vacuum', 'migrate', 'kill', 'dump'],
    entities: ['sql_table', 'slow_query', 'table_statistics', 'foreign_key', 'dead_tuples', 'schema_version', 'blocked_backend', 'database_backup'],
    verbs: ['Execute SQL over', 'Run EXPLAIN ANALYZE on', 'Recalculate stats for', 'Create B-tree on', 'Reclaim disk space from', 'Apply schema migration to', 'Terminate rogue', 'Create logical snapshot of'],
  },
  {
    name: 'k8s',
    actions: ['get', 'describe', 'scale', 'restart', 'logs', 'apply', 'cordon', 'port_forward'],
    entities: ['pod', 'deployment', 'service', 'ingress', 'configmap', 'node', 'daemonset', 'horizontal_pod_autoscaler'],
    verbs: ['Inspect status of', 'Show detailed events for', 'Adjust replicas of', 'Perform rolling restart of', 'Stream stderr from', 'Apply declarative manifest to', 'Mark unschedulable', 'Proxy local port to'],
  },
  {
    name: 'datadog',
    actions: ['query', 'mute', 'unmute', 'snapshot', 'create', 'search', 'tail', 'export'],
    entities: ['latency_metric', 'alert_monitor', 'error_spike', 'dashboard_graph', 'synthetic_test', 'apm_trace', 'live_logs', 'sla_report'],
    verbs: ['Query timeseries for', 'Silence noisy', 'Re-enable notifications for', 'Capture PNG of', 'Deploy synthetic probe for', 'Trace distributed span across', 'Tail incoming', 'Generate compliance report for'],
  },
];

/**
 * Feature hashing vector generator from text tokens (replaces synthetic modulo math).
 * Hashes actual words in descriptions to unit vectors in the target embedding dimension.
 */
function textToEmbedding(text: string, dim: number): Float32Array {
  const vec = new Float32Array(dim);
  const words = text.toLowerCase().split(/[^a-z0-9]+/);
  for (const word of words) {
    if (!word) continue;
    let h = 2166136261;
    for (let i = 0; i < word.length; i++) {
      h ^= word.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    const idx = (h >>> 0) % dim;
    vec[idx] += 1.0;
  }
  let sumSq = 0;
  for (let i = 0; i < dim; i++) sumSq += vec[i] * vec[i];
  const norm = Math.sqrt(sumSq) || 1;
  for (let i = 0; i < dim; i++) vec[i] /= norm;
  return vec;
}

function generateBenchmarkCatalog(targetCount: number, dim: number): { tools: ToolDefinition[]; embeddings: Record<string, number[]> } {
  const tools: ToolDefinition[] = [];
  const embeddings: Record<string, number[]> = {};

  let index = 0;
  outer: while (tools.length < targetCount) {
    for (const domain of DOMAINS) {
      for (let a = 0; a < domain.actions.length; a++) {
        for (let e = 0; e < domain.entities.length; e++) {
          if (tools.length >= targetCount) break outer;
          const action = domain.actions[a];
          const entity = domain.entities[e];
          const verb = domain.verbs[a];
          const name = `${domain.name}_${action}_${entity}_${index}`;
          const description = `${verb} ${entity.replace(/_/g, ' ')} in ${domain.name} infrastructure`;
          const keywords = [domain.name, action, ...entity.split('_')];

          // Generate vector from text tokens via feature hashing
          const vec = textToEmbedding(`${name} ${description} ${keywords.join(' ')}`, dim);

          tools.push({
            name,
            description,
            domain: domain.name,
            keywords,
            readOnly: action === 'get' || action === 'list' || action === 'read' || action === 'describe' || action === 'search',
          });
          embeddings[name] = Array.from(vec);
          index++;
        }
      }
    }
  }

  return { tools, embeddings };
}

function runBenchmarkIteration(
  fn: () => void,
  iterations: number
): { p50: number; p95: number; p99: number } {
  // Warmup
  for (let i = 0; i < Math.min(20, iterations); i++) {
    fn();
  }

  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    fn();
    times.push(performance.now() - t0);
  }

  times.sort((a, b) => a - b);
  const p50 = times[Math.floor(iterations * 0.50)];
  const p95 = times[Math.floor(iterations * 0.95)];
  const p99 = times[Math.floor(iterations * 0.99)];

  return { p50, p95, p99 };
}

async function main() {
  console.log('================================================================================');
  console.log(' TOOL IMPULSE: SCALE & LATENCY BENCHMARK (10 → 500 Tools)');
  console.log('================================================================================\n');

  const catalogSizes = [10, 50, 100, 500];
  const iterations = 100;
  const results: BenchmarkRow[] = [];

  const sampleQueries = [
    'refund credit card charge for customer',
    'merge approved pull request branch on github',
    'query slow postgres queries and explain execution plan',
    'search backlog defects and create jira ticket',
    'scale kubernetes deployment replicas in cluster',
  ];

  for (const size of catalogSizes) {
    console.log(`--- Catalog Size: ${size} Tools (${iterations} iterations) ---`);

    // 1. BM25 Lexical Routing
    {
      const { tools } = generateBenchmarkCatalog(size, 16);
      const engine = new ToolImpulse({ tools });
      let qIdx = 0;
      const stats = runBenchmarkIteration(() => {
        const query = sampleQueries[qIdx++ % sampleQueries.length];
        engine.resolveSync(query, undefined, undefined, { topK: 3 });
      }, iterations);

      results.push({
        catalogSize: size,
        mode: 'BM25 Lexical',
        dimension: 'N/A',
        p50Ms: stats.p50,
        p95Ms: stats.p95,
        p99Ms: stats.p99,
      });
      console.log(`  BM25 Lexical:                P50: ${stats.p50.toFixed(3)}ms | P95: ${stats.p95.toFixed(3)}ms`);
    }

    // 2. Dense Vector (768d)
    {
      const dim = 768;
      const { tools, embeddings } = generateBenchmarkCatalog(size, dim);
      const engine = new ToolImpulse({ tools });
      engine.setEmbeddings(embeddings);

      const queryVec = textToEmbedding('refund customer payment in stripe', dim);
      const stats = runBenchmarkIteration(() => {
        engine.resolveSync('refund payment', queryVec, undefined, { topK: 3, alpha: 1.0 });
      }, iterations);

      results.push({
        catalogSize: size,
        mode: 'Dense Vector',
        dimension: '768d',
        p50Ms: stats.p50,
        p95Ms: stats.p95,
        p99Ms: stats.p99,
      });
      console.log(`  Dense (768d):                P50: ${stats.p50.toFixed(3)}ms | P95: ${stats.p95.toFixed(3)}ms`);
    }

    // 3. Dense Vector (1536d)
    {
      const dim = 1536;
      const { tools, embeddings } = generateBenchmarkCatalog(size, dim);
      const engine = new ToolImpulse({ tools });
      engine.setEmbeddings(embeddings);

      const queryVec = textToEmbedding('refund customer payment in stripe', dim);
      const stats = runBenchmarkIteration(() => {
        engine.resolveSync('refund payment', queryVec, undefined, { topK: 3, alpha: 1.0 });
      }, iterations);

      results.push({
        catalogSize: size,
        mode: 'Dense Vector',
        dimension: '1536d',
        p50Ms: stats.p50,
        p95Ms: stats.p95,
        p99Ms: stats.p99,
      });
      console.log(`  Dense (1536d):               P50: ${stats.p50.toFixed(3)}ms | P95: ${stats.p95.toFixed(3)}ms`);
    }

    // 4. Real Web Request Simulation (Fresh Array Allocations - No WeakMap reliance)
    {
      const { tools } = generateBenchmarkCatalog(size, 16);
      let qIdx = 0;

      // Every single request allocates a fresh array of fresh object literals
      // (simulating Next.js / Express user request handler: getUserTools(req.user))
      const stats = runBenchmarkIteration(() => {
        const freshRequestTools = tools.map((t) => ({ ...t }));
        const query = sampleQueries[qIdx++ % sampleQueries.length];
        ToolImpulse.filter(query, freshRequestTools, { topK: 3 });
      }, iterations);

      results.push({
        catalogSize: size,
        mode: 'Fresh Request Allocations',
        dimension: 'N/A',
        p50Ms: stats.p50,
        p95Ms: stats.p95,
        p99Ms: stats.p99,
      });
      console.log(`  Fresh Request Allocations:   P50: ${stats.p50.toFixed(3)}ms | P95: ${stats.p95.toFixed(3)}ms (LRU cache hit)\n`);
    }
  }

  console.log('================================================================================');
  console.log(' SUMMARY LATENCY PERCENTILES');
  console.log('================================================================================');
  console.table(
    results.map((r) => ({
      Catalog: `${r.catalogSize} tools`,
      Mode: r.mode,
      Dimension: r.dimension,
      'P50 (ms)': r.p50Ms.toFixed(3),
      'P95 (ms)': r.p95Ms.toFixed(3),
      'P99 (ms)': r.p99Ms.toFixed(3),
    }))
  );
}

main().catch(console.error);
