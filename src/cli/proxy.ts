import { spawn } from 'node:child_process';
import * as readline from 'node:readline';
import { ToolImpulseEngine } from '../core/engine.js';
import { McpToolDefinition, mcpToImpulseTool } from '../adapters/mcp.js';
import { ImpulseSessionState } from '../core/types.js';

export interface StdioMcpProxyOptions {
  topK?: number;
  maxTotal?: number;
  invariantTools?: string[];
  initialQuery?: string;
}

export class StdioMcpProxy {
  private engine: ToolImpulseEngine;
  private session: ImpulseSessionState = {
    recentToolNames: [],
    turnCount: 0,
  };
  private fullToolCatalog: McpToolDefinition[] = [];
  private topK: number;
  private maxTotal: number;
  private invariantTools: Set<string>;
  private activeQuery: string;

  constructor(options: StdioMcpProxyOptions = {}) {
    this.engine = new ToolImpulseEngine();
    this.topK = options.topK || 3;
    this.maxTotal = options.maxTotal || 10;
    this.invariantTools = new Set(options.invariantTools || []);
    this.activeQuery = options.initialQuery || '';
  }

  /**
   * Filter the full tool catalog down to invariant tools + top-K dynamic impulse tools.
   */
  public filterToolList(tools: McpToolDefinition[]): McpToolDefinition[] {
    if (tools.length <= this.maxTotal) {
      return tools;
    }

    // 1. Separate invariant tools
    const invariantMatches: McpToolDefinition[] = [];
    const dynamicCandidates: McpToolDefinition[] = [];

    for (const tool of tools) {
      if (this.invariantTools.has(tool.name)) {
        invariantMatches.push(tool);
      } else {
        dynamicCandidates.push(tool);
      }
    }

    // 2. Resolve top-K dynamic tools via Tool Impulse Engine
    const dynamicSlots = Math.max(1, this.maxTotal - invariantMatches.length);
    const impulseResult = this.engine.resolveSync(this.activeQuery, undefined, this.session, {
      topK: Math.min(this.topK, dynamicSlots),
    });

    const activeDynamicNames = new Set(impulseResult.tools.map((t) => t.name));
    const selectedDynamic = dynamicCandidates.filter((t) => activeDynamicNames.has(t.name));

    // 3. Combine invariant + selected dynamic tools (guaranteed <= maxTotal)
    const combined = [...invariantMatches, ...selectedDynamic];

    // Fallback: If combined < maxTotal, fill remaining slots from candidates
    if (combined.length < this.maxTotal && dynamicCandidates.length > 0) {
      const combinedNames = new Set(combined.map((t) => t.name));
      for (const candidate of dynamicCandidates) {
        if (combined.length >= this.maxTotal) break;
        if (!combinedNames.has(candidate.name)) {
          combined.push(candidate);
          combinedNames.add(candidate.name);
        }
      }
    }

    return combined;
  }

  public async start(command: string, args: string[]): Promise<void> {
    const child = spawn(command, args, {
      stdio: ['pipe', 'pipe', 'inherit'],
    });

    const clientReader = readline.createInterface({
      input: process.stdin,
      terminal: false,
    });

    const downstreamReader = readline.createInterface({
      input: child.stdout,
      terminal: false,
    });

    // Handle messages from MCP client (Claude Desktop / Cursor)
    clientReader.on('line', async (line) => {
      if (!line.trim()) return;
      try {
        const msg = JSON.parse(line);

        // Track tool executions to update hysteresis and Bayesian transition graph
        if (msg.method === 'tools/call' && msg.params?.name) {
          const toolName = msg.params.name;
          this.session.recentToolNames = [toolName, ...(this.session.recentToolNames || []).slice(0, 4)];
          this.session.turnCount = (this.session.turnCount || 0) + 1;
          this.engine.recordExecution([toolName]);

          // Emit notification to client that tools list changed based on state shift
          const listChangedNotification = JSON.stringify({
            jsonrpc: '2.0',
            method: 'notifications/tools/list_changed',
          });
          process.stdout.write(listChangedNotification + '\n');
        }

        // If client sends sampling or custom query context, update active query
        if (msg.params?.prompt || msg.params?.query) {
          this.activeQuery = msg.params.prompt || msg.params.query;
        }

        // Pass message downstream to wrapped server
        child.stdin.write(line + '\n');
      } catch {
        child.stdin.write(line + '\n');
      }
    });

    // Handle messages from downstream MCP server
    downstreamReader.on('line', async (line) => {
      if (!line.trim()) return;
      try {
        const msg = JSON.parse(line);

        // Intercept tools/list response: actively prune the tool array!
        if (msg.result?.tools && Array.isArray(msg.result.tools)) {
          this.fullToolCatalog = msg.result.tools;
          const impulseTools = this.fullToolCatalog.map(mcpToImpulseTool);
          this.engine.registerToolsSync(impulseTools);

          // Apply active filtering to enforce strict context budget envelope
          msg.result.tools = this.filterToolList(this.fullToolCatalog);
          process.stdout.write(JSON.stringify(msg) + '\n');
          return;
        }

        // Forward all other messages unmodified
        process.stdout.write(line + '\n');
      } catch {
        process.stdout.write(line + '\n');
      }
    });

    child.on('exit', (code) => {
      process.exit(code || 0);
    });
  }
}

// CLI entry point
export async function runCli(): Promise<void> {
  const rawArgs = process.argv.slice(2);

  if (rawArgs.length === 0 || rawArgs.includes('--help') || rawArgs.includes('-h')) {
    console.error(`
Tool Impulse Engine (TIE) - Local MCP Proxy
Unconscious Perceptual Reflex for Autonomous Agent Tool Retrieval

Usage:
  npx tool-impulse-proxy [options] -- <command> [args...]

Examples:
  npx tool-impulse-proxy -- npx -y @modelcontextprotocol/server-everything
  npx tool-impulse-proxy --top-k 3 --max-total 10 --invariant search,remember -- node ./my-server.js

Options:
  --top-k <n>         Maximum dynamic impulse tools mounted per turn (default: 3)
  --max-total <n>     Strict ceiling on total mounted tools (default: 10)
  --invariant <tools> Comma-separated list of tool names always mounted
  --query <text>      Initial prompt query context
  -h, --help          Show this help message
  -v, --version       Show version
    `);
    process.exit(0);
  }

  if (rawArgs.includes('--version') || rawArgs.includes('-v')) {
    console.log('tool-impulse 1.1.0');
    process.exit(0);
  }

  // Parse options
  let topK = 3;
  let maxTotal = 10;
  let invariantTools: string[] = [];
  let initialQuery = '';
  let commandArgsStartIndex = 0;

  for (let i = 0; i < rawArgs.length; i++) {
    if (rawArgs[i] === '--') {
      commandArgsStartIndex = i + 1;
      break;
    }
    if (rawArgs[i] === '--top-k' && rawArgs[i + 1]) {
      topK = parseInt(rawArgs[++i], 10) || 3;
    } else if (rawArgs[i] === '--max-total' && rawArgs[i + 1]) {
      maxTotal = parseInt(rawArgs[++i], 10) || 10;
    } else if (rawArgs[i] === '--invariant' && rawArgs[i + 1]) {
      invariantTools = rawArgs[++i].split(',').map((s) => s.trim());
    } else if (rawArgs[i] === '--query' && rawArgs[i + 1]) {
      initialQuery = rawArgs[++i];
    } else if (!rawArgs[i].startsWith('--')) {
      // Direct command without '--' separator
      commandArgsStartIndex = i;
      break;
    }
  }

  const downstreamArgs = rawArgs.slice(commandArgsStartIndex);
  if (downstreamArgs.length === 0) {
    console.error('Error: No downstream MCP command provided. See --help.');
    process.exit(1);
  }

  const [cmd, ...cmdArgs] = downstreamArgs;
  const proxy = new StdioMcpProxy({
    topK,
    maxTotal,
    invariantTools,
    initialQuery,
  });
  await proxy.start(cmd, cmdArgs);
}
