import { describe, it, expect } from 'vitest';
import { ToolImpulse } from '../src/core/engine.js';
import {
  createGeminiToolFilter,
  GeminiFunctionDeclaration,
  GeminiToolGroup,
} from '../src/adapters/gemini.js';
import {
  createGrokToolFilter,
  createOpenAiToolFilter,
  OpenAiFunctionTool,
  toOpenAITools,
  fromOpenAITools,
} from '../src/adapters/openai.js';
import {
  createCloudflareAiFilter,
  CloudflareAiTool,
} from '../src/adapters/cloudflare.js';
import {
  createAnthropicToolFilter,
  AnthropicTool,
} from '../src/adapters/anthropic.js';
import { GeminiEmbedder } from '../src/embedders/gemini.js';
import { CloudflareEmbedder } from '../src/embedders/cloudflare.js';

describe('Native Ecosystem Adapters', () => {
  describe('Google Gemini Adapter', () => {
    const geminiDeclarations: GeminiFunctionDeclaration[] = [
      {
        name: 'stripe_create_charge',
        description: 'Charge a credit card in Stripe for orders',
        parameters: { type: 'OBJECT', properties: { amount: { type: 'NUMBER' } } },
      },
      {
        name: 'stripe_list_invoices',
        description: 'Fetch billing invoices from Stripe',
        parameters: { type: 'OBJECT', properties: { customerId: { type: 'STRING' } } },
      },
      {
        name: 'jira_create_issue',
        description: 'Create a new bug report or ticket in Jira',
        parameters: { type: 'OBJECT', properties: { summary: { type: 'STRING' } } },
      },
      {
        name: 'slack_send_dm',
        description: 'Send direct message to teammate on Slack',
        parameters: { type: 'OBJECT', properties: { userId: { type: 'STRING' } } },
      },
    ];

    it('filters function declarations down to top-K', async () => {
      const engine = new ToolImpulse();
      const filter = createGeminiToolFilter(engine, { topK: 2 });

      const { declarations, result } = await filter.filterDeclarations(
        'Charge customer card for order 500',
        geminiDeclarations
      );

      expect(declarations.length).toBeLessThanOrEqual(2);
      expect(declarations[0].name).toBe('stripe_create_charge');
      expect(result.selectedNames).toContain('stripe_create_charge');
    });

    it('formats tools in exact Google GenAI schema ({ tools: [{ functionDeclarations }] })', async () => {
      const engine = new ToolImpulse();
      const filter = createGeminiToolFilter(engine, { topK: 1 });

      const { tools, result } = await filter.formatTools(
        'Report urgent bug in Jira',
        geminiDeclarations
      );

      expect(tools.length).toBe(1);
      expect(tools[0].functionDeclarations).toBeDefined();
      expect(tools[0].functionDeclarations?.length).toBe(1);
      expect(tools[0].functionDeclarations![0].name).toBe('jira_create_issue');
      expect(result.tools[0].name).toBe('jira_create_issue');
    });

    it('filters pre-grouped Gemini tool groups', async () => {
      const engine = new ToolImpulse();
      const filter = createGeminiToolFilter(engine, { topK: 1 });

      const groups: GeminiToolGroup[] = [
        { functionDeclarations: geminiDeclarations },
      ];

      const { tools } = await filter.filterTools('Send Slack message', groups);
      expect(tools.length).toBe(1);
      expect(tools[0].functionDeclarations?.length).toBe(1);
      expect(tools[0].functionDeclarations![0].name).toBe('slack_send_dm');
    });
  });

  describe('OpenAI & xAI Grok Adapter', () => {
    const grokTools: OpenAiFunctionTool[] = [
      {
        type: 'function',
        function: {
          name: 'stripe_refund_payment',
          description: 'Refund credit card charge in Stripe',
          parameters: { type: 'object', properties: { chargeId: { type: 'string' } } },
        },
      },
      {
        type: 'function',
        function: {
          name: 'github_merge_pr',
          description: 'Merge pull request on GitHub repository',
          parameters: { type: 'object', properties: { prNumber: { type: 'number' } } },
        },
      },
      {
        type: 'function',
        function: {
          name: 'slack_post_announcement',
          description: 'Post release announcement to Slack general channel',
          parameters: { type: 'object', properties: { message: { type: 'string' } } },
        },
      },
    ];

    it('filters OpenAI/Grok function tools for chat completions', async () => {
      const engine = new ToolImpulse();
      const filter = createGrokToolFilter(engine, { topK: 1 });

      const { tools, result } = await filter.filterTools('Refund customer payment', grokTools);
      expect(tools.length).toBe(1);
      expect(tools[0].type).toBe('function');
      expect(tools[0].function.name).toBe('stripe_refund_payment');
      expect(result.selectedNames).toContain('stripe_refund_payment');
    });

    it('exports createOpenAiToolFilter alias identical to createGrokToolFilter', async () => {
      const engine = new ToolImpulse();
      const filter = createOpenAiToolFilter(engine, { topK: 1 });

      const { tools } = await filter.filterTools('Merge PR #42', grokTools);
      expect(tools.length).toBe(1);
      expect(tools[0].function.name).toBe('github_merge_pr');
    });

    it('converts to and from OpenAI function tools cleanly', () => {
      const toolDefs = fromOpenAITools(grokTools);
      expect(toolDefs.length).toBe(3);
      expect(toolDefs[0].name).toBe('stripe_refund_payment');
      expect(toolDefs[0].description).toBe('Refund credit card charge in Stripe');

      const convertedBack = toOpenAITools(toolDefs);
      expect(convertedBack.length).toBe(3);
      expect(convertedBack[0].function.name).toBe('stripe_refund_payment');
      expect(convertedBack[0].type).toBe('function');
    });
  });

  describe('Cloudflare Workers & Workers AI Adapter', () => {
    const cfTools: CloudflareAiTool[] = [
      {
        name: 'db_query_users',
        description: 'Query Postgres user accounts by email',
        parameters: { type: 'object' },
      },
      {
        name: 'cf_kv_get',
        description: 'Fetch cached session from Cloudflare KV',
        parameters: { type: 'object' },
      },
    ];

    it('filters Cloudflare Workers AI tool specifications in-memory', async () => {
      const engine = new ToolImpulse();
      const filter = createCloudflareAiFilter(engine, { topK: 1 });

      const { tools, result } = await filter.filterTools('Get cached user session from KV', cfTools);
      expect(tools.length).toBe(1);
      expect(tools[0].name).toBe('cf_kv_get');
      expect(result.selectedNames).toContain('cf_kv_get');
    });
  });

  describe('Anthropic Claude Adapter', () => {
    const anthropicTools: AnthropicTool[] = [
      {
        name: 'stripe_list_invoices',
        description: 'Fetch billing invoices for a customer',
        input_schema: { type: 'object', properties: { customerId: { type: 'string' } } },
      },
      {
        name: 'jira_create_issue',
        description: 'Create issue in Jira tracking system',
        input_schema: { type: 'object', properties: { summary: { type: 'string' } } },
      },
    ];

    it('filters Anthropic Claude tools with input_schema preserved', async () => {
      const engine = new ToolImpulse();
      const filter = createAnthropicToolFilter(engine, { topK: 1 });

      const { tools, result } = await filter.filterTools('Find invoice for customer 10', anthropicTools);
      expect(tools.length).toBe(1);
      expect(tools[0].name).toBe('stripe_list_invoices');
      expect(tools[0].input_schema).toBeDefined();
      expect(result.selectedNames).toContain('stripe_list_invoices');
    });
  });

  describe('Native Embedders', () => {
    it('initializes GeminiEmbedder with text-embedding-004 defaults', () => {
      const embedder = new GeminiEmbedder({ apiKey: 'mock-key' });
      expect(embedder.dimension).toBe(768);
    });

    it('initializes CloudflareEmbedder with bge-small defaults and handles missing credentials', async () => {
      const embedder = new CloudflareEmbedder();
      expect(embedder.dimension).toBe(384);

      await expect(embedder.embedBatch(['test'])).rejects.toThrow(
        'CloudflareEmbedder requires either an `env.AI` binding or both `accountId` and `apiToken`'
      );
    });
  });
});
