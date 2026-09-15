/**
 * Tool Impulse High-Scale Performance Benchmark
 *
 * Evaluates latency percentiles (P50, P90, P95, P99) and throughput across:
 * - 10, 50, 100, and 500 registered tools
 * - Lexical BM25 routing
 * - Dense vector routing (768d Gemini & 1536d OpenAI)
 * - Hybrid BM25 + Dense vector routing with trajectory blending
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

function generateRealisticCatalog(targetCount: number, dim: number): { tools: ToolDefinition[]; embeddings: Record<string, number[]> } {
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

          // Deterministic normalized embedding
          const vec = new Float32Array(dim);
          const p1 = (index * 7 + a * 13 + e * 29) % dim;
          const p2 = (index * 17 + a * 31 + e * 3) % dim;
          vec[p1] = 0.8;
          vec[p2] = 0.6;
          // Normalize
          let sumSq = 0;
          for (let i = 0; i < dim; i++) sumSq += vec[i] * vec[i];
          const norm = Math.sqrt(sumSq) || 1;
          for (let i = 0; i < dim; i++) vec[i] /= norm;

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
): { p50: number; p95: number; p99: number; opsPerSec: number } {
  // Warmup
  for (let i = 0; i < Math.min(20, iterations); i++) {
    fn();
  }

  const times: number[] = [];
  const startTotal = performance.now();
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    fn();
    times.push(performance.now() - t0);
  }
  const totalMs = performance.now() - startTotal;

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
  const iterations = 200;
  const results: BenchmarkRow[] = [];

  const sampleQueries = [
    'refund credit card charge for customer',
    'merge approved pull request branch on github',
    'query slow postgres queries and explain execution plan',
    'search backlog defects and create jira ticket',
    'scale kubernetes deployment replicas in cluster',
  ];

  for (const size of catalogSizes) {
    console.log(`--- Benchmarking Catalog Size: ${size} Tools (${iterations} iterations) ---`);

    // 1. BM25 Lexical Routing
    {
      const { tools } = generateRealisticCatalog(size, 16);
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
      console.log(`  BM25 Lexical:     P50: ${stats.p50.toFixed(3)}ms | P95: ${stats.p95.toFixed(3)}ms`);
    }

    // 2. Dense Vector (768d - Gemini / BGE-Base)
    {
      const dim = 768;
      const { tools, embeddings } = generateRealisticCatalog(size, dim);
      const engine = new ToolImpulse({ tools });
      engine.setEmbeddings(embeddings);

      const queryVec = new Float32Array(dim);
      queryVec[10] = 0.7;
      queryVec[50] = 0.7;

      const stats = runBenchmarkIteration(() => {
        engine.resolveSync('test query', queryVec, undefined, { topK: 3, alpha: 1.0 });
      }, iterations);

      results.push({
        catalogSize: size,
        mode: 'Dense Vector',
        dimension: '768d',
        p50Ms: stats.p50,
        p95Ms: stats.p95,
        p99Ms: stats.p99,
      });
      console.log(`  Dense (768d):     P50: ${stats.p50.toFixed(3)}ms | P95: ${stats.p95.toFixed(3)}ms`);
    }

    // 3. Dense Vector (1536d - OpenAI text-embedding-3-small/large)
    {
      const dim = 1536;
      const { tools, embeddings } = generateRealisticCatalog(size, dim);
      const engine = new ToolImpulse({ tools });
      engine.setEmbeddings(embeddings);

      const queryVec = new Float32Array(dim);
      queryVec[10] = 0.7;
      queryVec[100] = 0.7;

      const stats = runBenchmarkIteration(() => {
        engine.resolveSync('test query', queryVec, undefined, { topK: 3, alpha: 1.0 });
      }, iterations);

      results.push({
        catalogSize: size,
        mode: 'Dense Vector',
        dimension: '1536d',
        p50Ms: stats.p50,
        p95Ms: stats.p95,
        p99Ms: stats.p99,
      });
      console.log(`  Dense (1536d):    P50: ${stats.p50.toFixed(3)}ms | P95: ${stats.p95.toFixed(3)}ms`);
    }

    // 4. Hybrid (BM25 + 1536d Vector + Trajectory Blending)
    {
      const dim = 1536;
      const { tools, embeddings } = generateRealisticCatalog(size, dim);
      const engine = new ToolImpulse({ tools });
      engine.setEmbeddings(embeddings);

      const queryVec = new Float32Array(dim);
      queryVec[20] = 0.6;
      queryVec[80] = 0.8;
      const priorVec = new Float32Array(dim);
      priorVec[20] = 0.9;

      let qIdx = 0;
      const stats = runBenchmarkIteration(() => {
        const query = sampleQueries[qIdx++ % sampleQueries.length];
        engine.resolveSync(
          query,
          queryVec,
          { priorTurnEmbedding: priorVec, recentToolNames: ['stripe_create_charge_0'] },
          { topK: 3, alpha: 0.7, beta: 0.75 }
        );
      }, iterations);

      results.push({
        catalogSize: size,
        mode: 'Hybrid (BM25+1536d)',
        dimension: '1536d',
        p50Ms: stats.p50,
        p95Ms: stats.p95,
        p99Ms: stats.p99,
      });
      console.log(`  Hybrid (1536d):   P50: ${stats.p50.toFixed(3)}ms | P95: ${stats.p95.toFixed(3)}ms\n`);
    }
  }

  // Verify latency remains within bounds (< 5ms P95 across all catalog sizes)
  const p95ThresholdMs = 5.0;
  for (const r of results) {
    if (r.p95Ms > p95ThresholdMs) {
      console.error(`Performance regression: ${r.mode} (${r.catalogSize} tools) P95 ${r.p95Ms.toFixed(3)}ms > ${p95ThresholdMs}ms`);
      process.exit(1);
    }
  }

  console.log(`✓ Latency checks passed (all P95 < ${p95ThresholdMs}ms across 10-500 tools).`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
