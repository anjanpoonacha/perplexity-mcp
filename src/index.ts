#!/usr/bin/env bun
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { mkdirSync, writeFileSync, readFileSync, readdirSync, renameSync } from "fs";
import { join } from "path";

// ============================================================================
// Configuration
// ============================================================================

const TODAY = new Date().toISOString().slice(0, 10);
const BASE_URL = process.env.PERPLEXITY_BASE_URL ?? "http://localhost:3030/v1";
const TIMEOUT_MS = Number(process.env.PERPLEXITY_TIMEOUT_MS ?? 900_000);
const API_KEY = process.env.PERPLEXITY_API_KEY ?? "";
const JOB_DIR = "/tmp/perplexity-jobs";

const RECENCY_VALUES = ["hour", "day", "week", "month"] as const;

// ============================================================================
// Types
// ============================================================================

interface Message {
  role: string;
  content: string;
}

interface Job {
  id: string;
  status: "pending" | "done" | "error";
  query: string;
  result?: string;
  error?: string;
  created: number;
  finished?: number;
}

// ============================================================================
// Job helpers
// ============================================================================

function ensureJobDir() {
  mkdirSync(JOB_DIR, { recursive: true });
}

function jobPath(id: string) {
  return join(JOB_DIR, `${id}.json`);
}

function writeJob(job: Job) {
  ensureJobDir();
  const tmp = jobPath(job.id) + ".tmp";
  writeFileSync(tmp, JSON.stringify(job, null, 2));
  renameSync(tmp, jobPath(job.id));
}

function readJob(id: string): Job | null {
  try {
    return JSON.parse(readFileSync(jobPath(id), "utf-8")) as Job;
  } catch {
    return null;
  }
}

function makeJobId(query: string): string {
  return query.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40);
}

// ============================================================================
// Validation
// ============================================================================

const ResponseSchema = z.object({
  choices: z
    .array(
      z.object({
        message: z.object({ content: z.string() }),
        finish_reason: z.string().optional(),
      })
    )
    .min(1),
  citations: z.array(z.string()).optional(),
  usage: z
    .object({
      prompt_tokens: z.number().optional(),
      completion_tokens: z.number().optional(),
      total_tokens: z.number().optional(),
    })
    .optional(),
  id: z.string().optional(),
  model: z.string().optional(),
});

function validateMessages(messages: unknown): asserts messages is Message[] {
  if (!Array.isArray(messages)) throw new Error("'messages' must be an array");
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (!m?.role || typeof m.role !== "string")
      throw new Error(`messages[${i}].role must be a string`);
    if (typeof m.content !== "string")
      throw new Error(`messages[${i}].content must be a string`);
  }
}

// ============================================================================
// Core
// ============================================================================

function buildSystemMessage(extra?: string): Message {
  const date = TODAY;
  let content = `Today's date is ${date}.`;
  if (extra) content += ` ${extra}`;
  return { role: "system", content };
}

function mergeSystemMessage(messages: Message[], systemMsg: Message): Message[] {
  if (messages.length > 0 && messages[0].role === "system") {
    return [
      { role: "system", content: `${systemMsg.content}\n\n${messages[0].content}` },
      ...messages.slice(1),
    ];
  }
  return [systemMsg, ...messages];
}

function stripThinkingTokens(content: string): string {
  return content.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

interface CompletionOptions {
  messages: Message[];
  model: string;
  stripThinking?: boolean;
  systemExtra?: string;
  recencyFilter?: string;
  searchDomainFilter?: string[];
  searchContextSize?: string;
  searchMode?: string;
  searchType?: string;
  searchLanguageFilter?: string[];
  enableSearchClassifier?: boolean;
  responseFormat?: Record<string, unknown>;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  languagePreference?: string;
  returnImages?: boolean;
  returnRelatedQuestions?: boolean;
}

async function performChatCompletion(opts: CompletionOptions): Promise<string> {
  const {
    messages, model, stripThinking = false, systemExtra,
    recencyFilter, searchDomainFilter, searchContextSize,
    searchMode, searchType, searchLanguageFilter,
    enableSearchClassifier,
    responseFormat, temperature, maxTokens, topP,
    frequencyPenalty, presencePenalty,
    languagePreference, returnImages, returnRelatedQuestions,
  } = opts;

  const systemMsg = buildSystemMessage(systemExtra);
  const finalMessages = mergeSystemMessage(messages, systemMsg);

  const body: Record<string, unknown> = { model, messages: finalMessages };
  if (recencyFilter && RECENCY_VALUES.includes(recencyFilter as (typeof RECENCY_VALUES)[number])) {
    body.search_recency_filter = recencyFilter;
  }
  if (searchDomainFilter?.length) body.search_domain_filter = searchDomainFilter;
  if (searchContextSize) body.web_search_options = { search_context_size: searchContextSize };
  if (searchMode) body.search_mode = searchMode;
  if (searchType) body.search_type = searchType;
  if (searchLanguageFilter?.length) body.search_language_filter = searchLanguageFilter;
  if (enableSearchClassifier !== undefined) body.enable_search_classifier = enableSearchClassifier;
  if (responseFormat) body.response_format = responseFormat;
  if (temperature !== undefined) body.temperature = temperature;
  if (maxTokens !== undefined) body.max_tokens = maxTokens;
  if (topP !== undefined) body.top_p = topP;
  if (frequencyPenalty !== undefined) body.frequency_penalty = frequencyPenalty;
  if (presencePenalty !== undefined) body.presence_penalty = presencePenalty;
  if (languagePreference) body.language_preference = languagePreference;
  if (returnImages !== undefined) body.return_images = returnImages;
  if (returnRelatedQuestions !== undefined) body.return_related_questions = returnRelatedQuestions;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  let response: Response;
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (API_KEY) headers["Authorization"] = `Bearer ${API_KEY}`;

    response = await fetch(`${BASE_URL}/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
      // Bun has a 5-minute idle socket timeout by default. sonar-deep-research sends no
      // streaming bytes during inference (~150–280s), so the idle timer fires before our
      // AbortController can. Disable it — our AbortController is the real deadline.
      // @ts-ignore Bun-specific option
      timeout: false,
    });
    clearTimeout(timeoutId);
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Perplexity API timeout after ${TIMEOUT_MS}ms`);
    }
    throw new Error(`Network error: ${error}`);
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Perplexity API ${response.status}: ${text}`);
  }

  let content: string;
  try {
    const data = ResponseSchema.parse(await response.json());
    content = data.choices[0].message.content;
    if (stripThinking) content = stripThinkingTokens(content);
    if (data.citations?.length) {
      content += "\n\nCitations:\n";
      data.citations.forEach((c, i) => (content += `[${i + 1}] ${c}\n`));
    }
  } catch (error) {
    throw new Error(`Invalid API response: ${error}`);
  }

  return content;
}

// ============================================================================
// MCP Tool definitions
// ============================================================================

const SEARCH_PROPS_OPTIONAL = {
  search_context_size: { type: "string", enum: ["low", "medium", "high"], description: "Search depth. low=fast, high=exhaustive." },
  search_mode: { type: "string", enum: ["web", "academic", "sec"], description: "Source mode: web (default), academic, sec." },
  search_type: { type: "string", enum: ["fast", "pro", "auto"], description: "Search thoroughness." },
  search_language_filter: { type: "array", items: { type: "string" }, description: "ISO 639-1 language codes to filter results." },
} as const;

const MESSAGES_PROP = {
  type: "array",
  description: "Conversation history (alternative to query).",
  items: {
    type: "object",
    properties: { role: { type: "string" }, content: { type: "string" } },
    required: ["role", "content"],
  },
} as const;

const TOOLS = [
  {
    name: "perplexity_search",
    description: `Web search restricted to specified domains (sonar). Today is ${TODAY} — always include this date in queries. Requires domain(s) to focus results.`,
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: `Search query — specific, entity-rich. Always include today's date (${TODAY}) in the query.` },
        search_domain_filter: {
          type: "array",
          items: { type: "string" },
          description: 'Required. Domain allowlist/blocklist. Prefix with "-" to exclude. E.g. ["matomo.org"] or ["-reddit.com"].',
        },
        recency_filter: { type: "string", enum: ["hour", "day", "week", "month"], description: "Limit to recent results." },
        ...SEARCH_PROPS_OPTIONAL,
      },
      required: ["query", "search_domain_filter"],
    },
  },
  {
    name: "perplexity_research",
    description: `Deep research across hundreds of sources (sonar-deep-research). Today is ${TODAY} — always include this date in queries. Thorough report output. Takes up to 15min — returns a job ID immediately. Use perplexity_job_status to poll and perplexity_job_get to retrieve the result.`,
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: `Research question — specify dimensions, cap scope. Always include today's date (${TODAY}) in the query.` },
        messages: MESSAGES_PROP,
        strip_thinking: { type: "boolean", description: "Strip <think> tags to save tokens.", default: false },
        ...SEARCH_PROPS_OPTIONAL,
        search_domain_filter: { type: "array", items: { type: "string" }, description: 'Optional domain filter. Prefix with "-" to exclude.' },
      },
      required: ["query"],
    },
  },
  {
    name: "perplexity_reason",
    description: `Analytical reasoning with web grounding (sonar-pro). Today is ${TODAY} — always include this date in queries. Use for tradeoffs, decisions, pros/cons. Strips <think> tags automatically.`,
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: `Question or decision to reason about. Always include today's date (${TODAY}) in the query.` },
        messages: MESSAGES_PROP,
        response_format: { type: "object", description: 'JSON schema output: { "type": "json_schema", "json_schema": {...} }' },
        ...SEARCH_PROPS_OPTIONAL,
        search_domain_filter: { type: "array", items: { type: "string" }, description: 'Optional domain filter. Prefix with "-" to exclude.' },
      },
      required: ["query"],
    },
  },
  {
    name: "perplexity_job_status",
    description: "Check the status of an async perplexity_research job. Returns: pending | done | error.",
    inputSchema: {
      type: "object",
      properties: {
        job_id: { type: "string", description: "Job ID returned by perplexity_research." },
      },
      required: ["job_id"],
    },
  },
  {
    name: "perplexity_job_get",
    description: "Retrieve the result of a completed perplexity_research job. Returns the full research report. Errors if the job is still pending or failed.",
    inputSchema: {
      type: "object",
      properties: {
        job_id: { type: "string", description: "Job ID returned by perplexity_research." },
      },
      required: ["job_id"],
    },
  },
] as const;

// ============================================================================
// Server
// ============================================================================

const server = new Server(
  { name: "perplexity-mcp", version: "2.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const a = (args ?? {}) as Record<string, unknown>;

  function sharedOpts(): Partial<CompletionOptions> {
    return {
      searchDomainFilter: Array.isArray(a.search_domain_filter) ? a.search_domain_filter as string[] : undefined,
      searchContextSize: a.search_context_size as string | undefined,
      searchMode: a.search_mode as string | undefined,
      searchType: a.search_type as string | undefined,
      searchLanguageFilter: Array.isArray(a.search_language_filter) ? a.search_language_filter as string[] : undefined,
      enableSearchClassifier: typeof a.enable_search_classifier === "boolean" ? a.enable_search_classifier : undefined,
      responseFormat: a.response_format as Record<string, unknown> | undefined,
      temperature: typeof a.temperature === "number" ? a.temperature : undefined,
      maxTokens: typeof a.max_tokens === "number" ? a.max_tokens : undefined,
      topP: typeof a.top_p === "number" ? a.top_p : undefined,
      frequencyPenalty: typeof a.frequency_penalty === "number" ? a.frequency_penalty : undefined,
      presencePenalty: typeof a.presence_penalty === "number" ? a.presence_penalty : undefined,
      languagePreference: a.language_preference as string | undefined,
      returnImages: typeof a.return_images === "boolean" ? a.return_images : undefined,
      returnRelatedQuestions: typeof a.return_related_questions === "boolean" ? a.return_related_questions : undefined,
    };
  }

  function resolveMessages(): Message[] {
    if (a.query) return [{ role: "user", content: a.query as string }];
    if (a.messages) { validateMessages(a.messages); return a.messages; }
    throw new Error("Provide either 'query' or 'messages'");
  }

  try {
    let result: string;

    if (name === "perplexity_search") {
      if (!Array.isArray(a.search_domain_filter) || (a.search_domain_filter as string[]).length === 0) {
        throw new Error("search_domain_filter is required for perplexity_search");
      }
      result = await performChatCompletion({
        messages: [{ role: "user", content: a.query as string }],
        model: "sonar",
        recencyFilter: a.recency_filter as string | undefined,
        ...sharedOpts(),
      });

    } else if (name === "perplexity_research") {
      const id = makeJobId(query);
      const query = a.query as string ?? (resolveMessages()[0]?.content ?? "");
      const job: Job = { id, status: "pending", query, created: Date.now() };
      writeJob(job);

      // Spawn detached worker via CLI
      const cliPath = new URL("./cli.ts", import.meta.url).pathname;
      const proc = Bun.spawn(
        [process.execPath, cliPath, "research", query, "--job-id", id,
          ...(a.search_context_size ? ["--context-size", a.search_context_size as string] : []),
          ...(a.search_mode ? ["--mode", a.search_mode as string] : []),
          ...(a.search_type ? ["--type", a.search_type as string] : []),
          ...(a.strip_thinking ? ["--strip-thinking"] : []),
        ],
        {
          env: process.env as Record<string, string>,
          stdin: null,
          stdout: null,
          stderr: null,
          detached: true,
        }
      );
      proc.unref();

      result = `job:${id}  ${jobPath(id)}\n\nResearch started. Use perplexity_job_status to poll and perplexity_job_get to retrieve the result when done.`;

    } else if (name === "perplexity_reason") {
      result = await performChatCompletion({
        messages: resolveMessages(),
        model: "sonar-pro",
        stripThinking: true,
        systemExtra: "Think step-by-step. Evaluate all angles before concluding. Structure your response with clear sections.",
        ...sharedOpts(),
      });

    } else if (name === "perplexity_job_status") {
      const id = a.job_id as string;
      const job = readJob(id);
      if (!job) throw new Error(`job:${id} not found`);
      result = job.status;

    } else if (name === "perplexity_job_get") {
      const id = a.job_id as string;
      const job = readJob(id);
      if (!job) throw new Error(`job:${id} not found`);
      if (job.status === "pending") throw new Error(`job:${id} is still pending`);
      if (job.status === "error") throw new Error(`job:${id} failed: ${job.error}`);
      result = job.result!;

    } else {
      throw new Error(`Unknown tool: ${name}`);
    }

    return { content: [{ type: "text", text: result }] };
  } catch (error) {
    return {
      content: [{ type: "text", text: `Error: ${error instanceof Error ? error.message : error}` }],
      isError: true,
    };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
