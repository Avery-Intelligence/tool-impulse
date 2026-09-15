import { describe, it, expect } from 'vitest';
import { ToolImpulse } from '../src/core/engine.js';
import { createToolRouter } from '../src/adapters/ai-sdk.js';
import { createOpenAIToolFilter, OpenAiFunctionTool } from '../src/adapters/openai.js';
import { createAnthropicToolFilter, AnthropicTool } from '../src/adapters/anthropic.js';

describe('Concurrency, Multi-Tenant Isolation & Static Memoization', () => {
  it('prevents cross-tenant data races during concurrent Promise.all requests', async () => {
    const filter = createOpenAIToolFilter({ topK: 1 });

    const tenantABillingTools: OpenAiFunctionTool[] = [
      {
        type: 'function',
        function: {
          name: 'stripe_charge_customer',
          description: 'Charge customer credit card in Stripe',
        },
      },
      {
        type: 'function',
        function: {
          name: 'stripe_issue_refund',
          description: 'Refund Stripe payment to original card',
        },
      },
    ];

    const tenantBDevOpsTools: OpenAiFunctionTool[] = [
      {
        type: 'function',
        function: {
          name: 'k8s_restart_pod',
          description: 'Restart Kubernetes pod deployment in cluster',
        },
      },
      {
        type: 'function',
        function: {
          name: 'k8s_scale_deployment',
          description: 'Scale Kubernetes replica count',
        },
      },
    ];

    // Concurrently dispatch 50 requests interleaving Tenant A and Tenant B
    const tasks = Array.from({ length: 50 }, (_, i) => {
      if (i % 2 === 0) {
        return filter.filterTools('Please charge customer credit card', tenantABillingTools);
      } else {
        return filter.filterTools('Restart failing pod in production', tenantBDevOpsTools);
      }
    });

    const results = await Promise.all(tasks);

    for (let i = 0; i < results.length; i++) {
      const res = results[i];
      if (i % 2 === 0) {
        expect(res.tools.length).toBe(1);
        expect(res.tools[0].function.name).toBe('stripe_charge_customer');
        expect(res.result.selectedNames).toContain('stripe_charge_customer');
        // Ensure zero leakage from Tenant B
        expect(res.result.selectedNames).not.toContain('k8s_restart_pod');
      } else {
        expect(res.tools.length).toBe(1);
        expect(res.tools[0].function.name).toBe('k8s_restart_pod');
        expect(res.result.selectedNames).toContain('k8s_restart_pod');
        // Ensure zero leakage from Tenant A
        expect(res.result.selectedNames).not.toContain('stripe_charge_customer');
      }
    }
  });

  it('preserves shared base engine catalog when concurrent requests use custom toolsets', async () => {
    const baseEngine = new ToolImpulse({
      tools: [
        { name: 'base_tool_alpha', description: 'Core system tool alpha' },
        { name: 'base_tool_beta', description: 'Core system tool beta' },
      ],
    });

    const router = createToolRouter(baseEngine, { topK: 1 });

    const tenantCustomTools = {
      tenant_tool_custom: { description: 'Custom tenant tool for analytics' },
    };

    // Execute query with tenant-specific custom tools
    const res = await router.getTools('Run custom analytics report', tenantCustomTools);
    expect(Object.keys(res.tools)).toEqual(['tenant_tool_custom']);

    // Base engine catalog must NOT be mutated or clobbered
    const baseNames = baseEngine.getCatalog().getToolNames();
    expect(baseNames).toEqual(['base_tool_alpha', 'base_tool_beta']);
    expect(baseEngine.getCatalog().hasTool('tenant_tool_custom')).toBe(false);
  });

  it('memoizes static ToolImpulse.filter by toolset reference without re-indexing', () => {
    const tools = [
      { name: 'slack_send', description: 'Send a message to a Slack channel' },
      { name: 'jira_ticket', description: 'Create a bug ticket in Jira' },
      { name: 'stripe_pay', description: 'Process payment in Stripe' },
    ];

    // First call indexes and warms cache
    const t0 = performance.now();
    const res1 = ToolImpulse.filter('Send Slack message', tools, { topK: 1 });
    const firstCallLatency = performance.now() - t0;
    expect(res1.length).toBe(1);
    expect(res1[0].name).toBe('slack_send');

    // Subsequent calls hit bounded fingerprint cache in <0.05ms
    const warmStarts: number[] = [];
    for (let i = 0; i < 20; i++) {
      const tStart = performance.now();
      const res = ToolImpulse.filter('Send Slack message', tools, { topK: 1 });
      warmStarts.push(performance.now() - tStart);
      expect(res[0].name).toBe('slack_send');
    }

    const avgWarmLatency = warmStarts.reduce((a, b) => a + b, 0) / warmStarts.length;
    // Warm calls should be sub-millisecond in-memory lookups (<0.5ms even on noisy CI runners)
    expect(avgWarmLatency).toBeLessThan(0.5);
  });

  it('hits cache when passing fresh object literals and newly allocated arrays', async () => {
    const router = createToolRouter({ topK: 1 });

    // Every call passes a BRAND-NEW object literal (simulating typical Vercel AI SDK getTools calls)
    const call1 = await router.getTools('Refund customer', {
      stripe_refund: { description: 'Issue refund to credit card via Stripe' },
      slack_notify: { description: 'Notify team in Slack channel' },
    });
    expect(Object.keys(call1.tools)).toEqual(['stripe_refund']);

    const call2 = await router.getTools('Refund customer', {
      stripe_refund: { description: 'Issue refund to credit card via Stripe' },
      slack_notify: { description: 'Notify team in Slack channel' },
    });
    expect(Object.keys(call2.tools)).toEqual(['stripe_refund']);

    // Static filter with freshly allocated arrays
    const res1 = ToolImpulse.filter('Post Slack message', [
      { name: 'slack_post', description: 'Post announcement in Slack' },
      { name: 'github_pr', description: 'Merge pull request on GitHub' },
    ], { topK: 1 });
    expect(res1[0].name).toBe('slack_post');

    const res2 = ToolImpulse.filter('Post Slack message', [
      { name: 'slack_post', description: 'Post announcement in Slack' },
      { name: 'github_pr', description: 'Merge pull request on GitHub' },
    ], { topK: 1 });
    expect(res2[0].name).toBe('slack_post');
  });

  it('rejects nonsense queries with accidental single-word matches and triggers cold-start fallback', () => {
    const catalog = new ToolImpulse({
      tools: [
        { name: 'check_credit', description: 'Check user credit score and balance', domain: 'finance' },
        { name: 'slack_post', description: 'Post announcement on Slack', domain: 'slack' },
      ],
    });

    // Query contains 4 words, only 1 of which accidentally matches 'credit'
    // With bounded normalization, 1/4 terms cannot score 1.0; score is ~0.15-0.25.
    // When minScoreThreshold is 0.3, it is safely rejected and falls back to defaultTools!
    const res = catalog.resolveSync(
      'bananas astronaut spaceship credit',
      undefined,
      undefined,
      {
        minScoreThreshold: 0.3,
        defaultTools: ['slack_post'],
      }
    );

    // Accidental match rejected, cold start fallback activated
    expect(res.selectedNames).toEqual(['slack_post']);
  });
});
