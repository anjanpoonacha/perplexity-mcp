#!/usr/bin/env bun
/**
 * perplexity-cli — CLI wrapper around Perplexity API
 *
 * Commands:
 *   search <query> <domain>   — Web search restricted to domain (sonar)
 *   research <query>          — Deep research report, async by default (sonar-deep-research)
 *   reason <query>            — Analytical reasoning with web grounding (sonar-pro)
 *   job status <id>           — Check job status: pending | done | error
 *   job get <id>              — Print job result (polls until done)
 *   job list                  — List all jobs
 *
 * Environment:
 *   PERPLEXITY_BASE_URL   — API base URL (default: http://localhost:3030/v1)
 *   PERPLEXITY_API_KEY    — API key
 *   PERPLEXITY_TIMEOUT_MS — Timeout in ms (default: 900000)
 */

import { z } from "zod";
import { mkdirSync, writeFileSync, readFileSync, readdirSync, renameSync } from "fs";
import { join } from "path";

// ============================================================================
// Configuration
// ============================================================================

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

function listJobs(): Job[] {
  ensureJobDir();
  return readdirSync(JOB_DIR)
    .filter((f) => f.endsWith(".json"))
    .map((f) => {
      try { return JSON.parse(readFileSync(join(JOB_DIR, f), "utf-8")) as Job; } catch { return null; }
    })
    .filter(Boolean) as Job[];
}

function makeJobId(query: string): string {
  return query.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40);
}

// ============================================================================
// Core
// ============================================================================

function buildSystemMessage(extra?: string): Message {
  const date = new Date().toISOString().slice(0, 10);
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
  // Search params
  recencyFilter?: string;
  searchDomainFilter?: string[];
  searchContextSize?: string;
  searchMode?: string;
  searchType?: string;
  searchLanguageFilter?: string[];
  // Generation params
  responseFormat?: Record<string, unknown>;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  // Response enrichment
  languagePreference?: string;
  returnImages?: boolean;
  returnRelatedQuestions?: boolean;
  // CLI-only
  rawJson?: boolean;
}

async function performChatCompletion(opts: CompletionOptions): Promise<void> {
  const {
    messages, model, stripThinking = false, systemExtra,
    recencyFilter, searchDomainFilter, searchContextSize,
    searchMode, searchType, searchLanguageFilter,
    responseFormat, temperature, maxTokens, topP,
    frequencyPenalty, presencePenalty,
    languagePreference, returnImages, returnRelatedQuestions,
    rawJson = false,
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

    if (!rawJson) body.stream = true;

    response = await fetch(`${BASE_URL}/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
      // @ts-ignore Bun-specific: disable idle socket timeout
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

  if (rawJson) {
    const json = await response.json();
    process.stdout.write(JSON.stringify(json, null, 2) + "\n");
    return;
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let citations: string[] = [];
  let thinkOpen = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    const lines = buf.split("\n");
    buf = lines.pop()!;

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") continue;

      let chunk: Record<string, unknown>;
      try { chunk = JSON.parse(data); } catch { continue; }

      if (Array.isArray(chunk.citations)) citations = chunk.citations as string[];

      const delta = (chunk as any).choices?.[0]?.delta?.content;
      if (typeof delta !== "string" || delta === "") continue;

      if (stripThinking) {
        let text = delta;
        if (thinkOpen) {
          const end = text.indexOf("</think>");
          if (end === -1) continue;
          text = text.slice(end + 8);
          thinkOpen = false;
        }
        const start = text.indexOf("<think>");
        if (start !== -1) {
          process.stdout.write(text.slice(0, start));
          thinkOpen = true;
          continue;
        }
        process.stdout.write(text);
      } else {
        process.stdout.write(delta);
      }
    }
  }

  process.stdout.write("\n");

  if (citations.length) {
    process.stdout.write("\nCitations:\n");
    citations.forEach((c, i) => process.stdout.write(`[${i + 1}] ${c}\n`));
  }
}

// Non-streaming variant that returns a string (used by async job worker)
async function performChatCompletionString(opts: CompletionOptions): Promise<string> {
  const {
    messages, model, stripThinking = false, systemExtra,
    recencyFilter, searchDomainFilter, searchContextSize,
    searchMode, searchType, searchLanguageFilter,
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
      // @ts-ignore Bun-specific
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

  const ResponseSchemaLocal = z.object({
    choices: z.array(z.object({ message: z.object({ content: z.string() }) })).min(1),
    citations: z.array(z.string()).optional(),
  });

  const data = ResponseSchemaLocal.parse(await response.json());
  let content = data.choices[0].message.content;
  if (stripThinking) content = stripThinkingTokens(content);
  if (data.citations?.length) {
    content += "\n\nCitations:\n";
    data.citations.forEach((c, i) => (content += `[${i + 1}] ${c}\n`));
  }
  return content;
}

// ============================================================================
// CLI
// ============================================================================

const CONTEXT_SIZE_VALUES = ["low", "medium", "high"] as const;
const SEARCH_MODE_VALUES = ["web", "academic", "sec"] as const;
const SEARCH_TYPE_VALUES = ["fast", "pro", "auto"] as const;

const HELP = `\x1b[1mperplexity\x1b[0m — Search, research & reason via Perplexity API

\x1b[1mUSAGE:\x1b[0m
  perplexity <command> [args] [options]

\x1b[1mCOMMANDS:\x1b[0m
  search <query> <domain>   Web search restricted to domain (sonar)
                            domain: comma-separated, prefix "-" to exclude
  research <query>          Deep research report (sonar-deep-research, up to 15min)
                            Async by default — returns job ID immediately
  reason <query>            Analytical reasoning with web grounding (sonar-pro)
                            Strips <think> tags automatically
  job status <id>           Check job status: pending | done | error
  job get <id>              Print job result (polls until done)
  job list                  List all jobs with status

\x1b[1mSEARCH OPTIONS:\x1b[0m
  --recency <hour|day|week|month>  Limit to recent results (search only)
  --context-size <low|medium|high> Search depth vs cost (high = thorough, low = fast)
  --mode <web|academic|sec>        Search source (web, academic papers, SEC filings)
  --type <fast|pro|auto>           Search type (fast = quick, pro = deep)
  --lang <codes>                   Comma-separated ISO 639-1 language codes

\x1b[1mGENERATION OPTIONS:\x1b[0m
  --temperature <float>            Sampling temperature (0 = deterministic)
  --max-tokens <int>               Maximum response tokens
  --top-p <float>                  Nucleus sampling (0-1, lower = more focused)
  --frequency-penalty <float>      Penalize repeated tokens (cannot combine with --presence-penalty)
  --presence-penalty <float>       Penalize tokens already used (cannot combine with --frequency-penalty)

\x1b[1mOUTPUT OPTIONS:\x1b[0m
  --async false                    Block until research completes (research only)
  --language <code>                Preferred response language (ISO 639-1, e.g. "en", "fr")
  --images                         Include image results
  --related                        Include related follow-up questions
  --strip-thinking                 Remove <think> tags (auto-enabled for reason)
  --model <model>                  Override the default model
  --json                           Output raw JSON response
  -h, --help                       Show this help

\x1b[1mPROMPTING TIPS:\x1b[0m
  - Always include today's date in the query for temporal context:
      "As of $(date +%Y-%m-%d), what is the latest..."
  - Be specific — the query seeds the web search directly.
  - Cap lists: "top 5" not "as many as possible".
  - Use domain arg for site restrictions (prose like "only search X" is ignored).

\x1b[1mENVIRONMENT:\x1b[0m
  PERPLEXITY_BASE_URL    API base URL (default: http://localhost:3030/v1)
  PERPLEXITY_API_KEY     API key
  PERPLEXITY_TIMEOUT_MS  Timeout in ms (default: 900000)

\x1b[1mEXAMPLES:\x1b[0m
  perplexity search "matomo archiving" matomo.org
  perplexity search "matomo archiving" "matomo.org,github.com"
  perplexity search "quantum error correction" --mode academic --context-size high
  perplexity research "state of AI agents as of $(date +%Y-%m-%d)" --context-size high
  perplexity research "deep topic" --async false
  perplexity job status a1b2c3d4
  perplexity job get a1b2c3d4
  perplexity reason "Rust or Go for a CLI tool?" --temperature 0.2
`;

function parseArgs(argv: string[]) {
  const args = argv.slice(2);

  if (args.length === 0 || args.includes("-h") || args.includes("--help")) {
    console.log(HELP);
    process.exit(0);
  }

  const command = args[0];
  const validCommands = ["search", "research", "reason", "job"];
  if (!validCommands.includes(command)) {
    console.error(`\x1b[31mError:\x1b[0m Unknown command "${command}"\n`);
    console.error(`Valid commands: ${validCommands.join(", ")}`);
    console.error(`Run "perplexity --help" for usage.`);
    process.exit(1);
  }

  // job subcommands handled separately
  if (command === "job") {
    return { command: "job", jobArgs: args.slice(1) } as any;
  }

  let query = "";
  let domain: string[] | undefined;
  let recency: string | undefined;
  let contextSize: string | undefined;
  let searchMode: string | undefined;
  let searchType: string | undefined;
  let searchLangs: string[] | undefined;
  let temperature: number | undefined;
  let maxTokens: number | undefined;
  let topP: number | undefined;
  let frequencyPenalty: number | undefined;
  let presencePenalty: number | undefined;
  let languagePreference: string | undefined;
  let returnImages = false;
  let returnRelatedQuestions = false;
  let stripThinking = false;
  let model: string | undefined;
  let rawJson = false;
  let async_ = true; // research default
  let jobId: string | undefined; // hidden internal flag

  const queryParts: string[] = [];
  let domainParsed = false;

  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--recency" && i + 1 < args.length) {
      recency = args[++i];
      if (!RECENCY_VALUES.includes(recency as any)) {
        console.error(`\x1b[31mError:\x1b[0m Invalid recency "${recency}". Use: ${RECENCY_VALUES.join(", ")}`);
        process.exit(1);
      }
    } else if (arg === "--context-size" && i + 1 < args.length) {
      contextSize = args[++i];
      if (!CONTEXT_SIZE_VALUES.includes(contextSize as any)) {
        console.error(`\x1b[31mError:\x1b[0m Invalid context-size "${contextSize}". Use: ${CONTEXT_SIZE_VALUES.join(", ")}`);
        process.exit(1);
      }
    } else if (arg === "--mode" && i + 1 < args.length) {
      searchMode = args[++i];
      if (!SEARCH_MODE_VALUES.includes(searchMode as any)) {
        console.error(`\x1b[31mError:\x1b[0m Invalid mode "${searchMode}". Use: ${SEARCH_MODE_VALUES.join(", ")}`);
        process.exit(1);
      }
    } else if (arg === "--type" && i + 1 < args.length) {
      searchType = args[++i];
      if (!SEARCH_TYPE_VALUES.includes(searchType as any)) {
        console.error(`\x1b[31mError:\x1b[0m Invalid type "${searchType}". Use: ${SEARCH_TYPE_VALUES.join(", ")}`);
        process.exit(1);
      }
    } else if (arg === "--lang" && i + 1 < args.length) {
      searchLangs = args[++i].split(",").map((l) => l.trim()).filter(Boolean);
    } else if (arg === "--temperature" && i + 1 < args.length) {
      temperature = parseFloat(args[++i]);
      if (isNaN(temperature)) { console.error(`\x1b[31mError:\x1b[0m --temperature must be a number`); process.exit(1); }
    } else if (arg === "--max-tokens" && i + 1 < args.length) {
      maxTokens = parseInt(args[++i], 10);
      if (isNaN(maxTokens)) { console.error(`\x1b[31mError:\x1b[0m --max-tokens must be an integer`); process.exit(1); }
    } else if (arg === "--top-p" && i + 1 < args.length) {
      topP = parseFloat(args[++i]);
      if (isNaN(topP)) { console.error(`\x1b[31mError:\x1b[0m --top-p must be a number`); process.exit(1); }
    } else if (arg === "--frequency-penalty" && i + 1 < args.length) {
      frequencyPenalty = parseFloat(args[++i]);
      if (isNaN(frequencyPenalty)) { console.error(`\x1b[31mError:\x1b[0m --frequency-penalty must be a number`); process.exit(1); }
    } else if (arg === "--presence-penalty" && i + 1 < args.length) {
      presencePenalty = parseFloat(args[++i]);
      if (isNaN(presencePenalty)) { console.error(`\x1b[31mError:\x1b[0m --presence-penalty must be a number`); process.exit(1); }
    } else if (arg === "--language" && i + 1 < args.length) {
      languagePreference = args[++i];
    } else if (arg === "--async" && i + 1 < args.length) {
      async_ = args[++i] !== "false";
    } else if (arg === "--images") {
      returnImages = true;
    } else if (arg === "--related") {
      returnRelatedQuestions = true;
    } else if (arg === "--strip-thinking") {
      stripThinking = true;
    } else if (arg === "--model" && i + 1 < args.length) {
      model = args[++i];
    } else if (arg === "--json") {
      rawJson = true;
    } else if (arg === "--job-id" && i + 1 < args.length) {
      jobId = args[++i];
    } else if (arg.startsWith("-")) {
      console.error(`\x1b[31mError:\x1b[0m Unknown option "${arg}"`);
      process.exit(1);
    } else {
      // positional: first non-flag = query parts, for search second positional = domain
      if (command === "search" && queryParts.length > 0 && !domainParsed) {
        domain = arg.split(",").map((d) => d.trim()).filter(Boolean);
        domainParsed = true;
      } else {
        queryParts.push(arg);
      }
    }
  }

  query = queryParts.join(" ");

  if (command === "search" && !domain) {
    console.error(`\x1b[31mError:\x1b[0m search requires a domain argument.`);
    console.error(`Usage: perplexity search "query" domain.com`);
    process.exit(1);
  }

  return {
    command, query, domain, recency, contextSize, searchMode, searchType,
    searchLangs, temperature, maxTokens, topP, frequencyPenalty, presencePenalty,
    languagePreference, returnImages, returnRelatedQuestions, stripThinking,
    model, rawJson, async_, jobId,
  };
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf-8").trim();
}

// ============================================================================
// Job commands
// ============================================================================

function fmtAge(ts: number): string {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

async function runJobCommand(jobArgs: string[]) {
  const sub = jobArgs[0];

  if (sub === "status") {
    const id = jobArgs[1];
    if (!id) { console.error("Usage: perplexity job status <id>"); process.exit(1); }
    const job = readJob(id);
    if (!job) { console.error(`job:${id} not found`); process.exit(1); }
    console.log(job.status);

  } else if (sub === "get") {
    const id = jobArgs[1];
    if (!id) { console.error("Usage: perplexity job get <id>"); process.exit(1); }
    // poll until done
    while (true) {
      const job = readJob(id);
      if (!job) { console.error(`job:${id} not found`); process.exit(1); }
      if (job.status === "done") { process.stdout.write(job.result! + "\n"); break; }
      if (job.status === "error") { console.error(`job:${id} failed: ${job.error}`); process.exit(1); }
      await Bun.sleep(3000);
    }

  } else if (sub === "list") {
    const jobs = listJobs().sort((a, b) => b.created - a.created);
    if (!jobs.length) { console.log("No jobs found."); return; }
    const statusColor = (s: string) =>
      s === "done" ? `\x1b[32m${s}\x1b[0m` : s === "error" ? `\x1b[31m${s}\x1b[0m` : `\x1b[33m${s}\x1b[0m`;
    for (const j of jobs) {
      const preview = j.query.slice(0, 60) + (j.query.length > 60 ? "…" : "");
      console.log(`${j.id}  ${statusColor(j.status).padEnd(20)}  ${fmtAge(j.created).padEnd(10)}  ${preview}`);
    }

  } else {
    console.error(`Unknown job subcommand "${sub}". Use: status | get | list`);
    process.exit(1);
  }
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const parsed = parseArgs(process.argv);

  if (parsed.command === "job") {
    await runJobCommand(parsed.jobArgs);
    return;
  }

  const {
    command, domain, recency, contextSize, searchMode, searchType, searchLangs,
    temperature, maxTokens, topP, frequencyPenalty, presencePenalty,
    languagePreference, returnImages, returnRelatedQuestions,
    stripThinking, model, rawJson, async_, jobId,
  } = parsed;

  let query = parsed.query;
  if (!query && !process.stdin.isTTY) query = await readStdin();
  if (!query) { console.error(`\x1b[31mError:\x1b[0m No query provided.`); process.exit(1); }

  const modelMap: Record<string, string> = {
    search: "sonar",
    research: "sonar-deep-research",
    reason: "sonar-pro",
  };

  const systemExtraMap: Record<string, string | undefined> = {
    search: undefined,
    research: undefined,
    reason: "Think step-by-step. Evaluate all angles before concluding. Structure your response with clear sections.",
  };

  const completionOpts: CompletionOptions = {
    messages: [{ role: "user", content: query }],
    model: model ?? modelMap[command],
    stripThinking: stripThinking || command === "reason",
    systemExtra: systemExtraMap[command],
    recencyFilter: recency,
    searchDomainFilter: domain,
    searchContextSize: contextSize,
    searchMode,
    searchType,
    searchLanguageFilter: searchLangs,
    temperature,
    maxTokens,
    topP,
    frequencyPenalty,
    presencePenalty,
    languagePreference,
    returnImages: returnImages || undefined,
    returnRelatedQuestions: returnRelatedQuestions || undefined,
    rawJson,
  };

  // Internal: running as background job worker
  if (jobId) {
    const job: Job = { id: jobId, status: "pending", query, created: Date.now() };
    writeJob(job);
    try {
      const result = await performChatCompletionString(completionOpts);
      writeJob({ ...job, status: "done", result, finished: Date.now() });
    } catch (err) {
      writeJob({ ...job, status: "error", error: err instanceof Error ? err.message : String(err), finished: Date.now() });
    }
    return;
  }

  // research: async by default
  if (command === "research" && async_) {
    const id = makeJobId(query);
    const job: Job = { id, status: "pending", query, created: Date.now() };
    writeJob(job);

    // Spawn detached child
    const proc = Bun.spawn(
      [process.execPath, process.argv[1], "research", query, "--job-id", id,
        ...(model ? ["--model", model] : []),
        ...(contextSize ? ["--context-size", contextSize] : []),
        ...(searchMode ? ["--mode", searchMode] : []),
        ...(searchType ? ["--type", searchType] : []),
        ...(recency ? ["--recency", recency] : []),
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

    console.log(`job:${id}  ${jobPath(id)}`);
    return;
  }

  try {
    await performChatCompletion(completionOpts);
  } catch (error) {
    console.error(`\x1b[31mError:\x1b[0m ${error instanceof Error ? error.message : error}`);
    process.exit(1);
  }
}

main();
