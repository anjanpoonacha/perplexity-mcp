#!/usr/bin/env bun
/**
 * perplexity-cli — CLI wrapper around Perplexity API
 *
 * Commands:
 *   search <query>    — Web search with citations (sonar)
 *   ask <query>       — Multi-turn conversation (sonar-pro)
 *   research <query>  — Deep research across hundreds of sources (sonar-deep-research)
 *   reason <query>    — Analytical reasoning with web grounding (sonar-pro)
 *
 * Options:
 *   --recency <hour|day|week|month>  — Limit results to recent timeframe (search only)
 *   --strip-thinking                 — Remove <think> tags from response (research/reason)
 *   --model <model>                  — Override the default model
 *   --json                           — Output raw JSON response
 *   --help                           — Show help
 *
 * Environment:
 *   PERPLEXITY_BASE_URL   — API base URL (default: http://localhost:3030/v1)
 *   PERPLEXITY_API_KEY    — API key
 *   PERPLEXITY_TIMEOUT_MS — Timeout in ms (default: 300000)
 */

import { z } from "zod";

// ============================================================================
// Configuration
// ============================================================================

const BASE_URL = process.env.PERPLEXITY_BASE_URL ?? "http://localhost:3030/v1";
const TIMEOUT_MS = Number(process.env.PERPLEXITY_TIMEOUT_MS ?? 300_000);
const API_KEY = process.env.PERPLEXITY_API_KEY ?? "";

const RECENCY_VALUES = ["hour", "day", "week", "month"] as const;

// ============================================================================
// Types
// ============================================================================

interface Message {
  role: string;
  content: string;
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

async function performChatCompletion(opts: CompletionOptions): Promise<string> {
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
  // Search params
  if (recencyFilter && RECENCY_VALUES.includes(recencyFilter as (typeof RECENCY_VALUES)[number])) {
    body.search_recency_filter = recencyFilter;
  }
  if (searchDomainFilter?.length) body.search_domain_filter = searchDomainFilter;
  if (searchContextSize) body.web_search_options = { search_context_size: searchContextSize };
  if (searchMode) body.search_mode = searchMode;
  if (searchType) body.search_type = searchType;
  if (searchLanguageFilter?.length) body.search_language_filter = searchLanguageFilter;

  // Generation params
  if (responseFormat) body.response_format = responseFormat;
  if (temperature !== undefined) body.temperature = temperature;
  if (maxTokens !== undefined) body.max_tokens = maxTokens;
  if (topP !== undefined) body.top_p = topP;
  if (frequencyPenalty !== undefined) body.frequency_penalty = frequencyPenalty;
  if (presencePenalty !== undefined) body.presence_penalty = presencePenalty;
  // Response enrichment
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

  const json = await response.json();

  if (rawJson) {
    return JSON.stringify(json, null, 2);
  }

  let content: string;
  try {
    const data = ResponseSchema.parse(json);
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
// CLI
// ============================================================================

const CONTEXT_SIZE_VALUES = ["low", "medium", "high"] as const;

const SEARCH_MODE_VALUES = ["web", "academic", "sec"] as const;
const SEARCH_TYPE_VALUES = ["fast", "pro", "auto"] as const;

const HELP = `\x1b[1mperplexity\x1b[0m — Search, ask, research & reason via Perplexity API

\x1b[1mUSAGE:\x1b[0m
  perplexity <command> <query> [options]

\x1b[1mCOMMANDS:\x1b[0m
  search <query>    Web search with citations (sonar)
  ask <query>       Multi-turn conversation (sonar-pro)
  research <query>  Deep research report (sonar-deep-research, 30-60s)
  reason <query>    Analytical reasoning with web grounding (sonar-pro)

\x1b[1mSEARCH OPTIONS:\x1b[0m
  --recency <hour|day|week|month>  Limit to recent results (search only)
  --domain <domains>               Comma-separated domains. Prefix "-" to exclude.
  --context-size <low|medium|high> Search depth vs cost (high = thorough, low = fast)
  --mode <web|academic|sec>        Search source (web, academic papers, SEC filings)
  --type <fast|pro|auto>           Search type (fast = quick, pro = deep)
  --lang <codes>                   Comma-separated ISO 639-1 language codes for search results

\x1b[1mGENERATION OPTIONS:\x1b[0m
  --temperature <float>            Sampling temperature (0 = deterministic)
  --max-tokens <int>               Maximum response tokens
  --top-p <float>                  Nucleus sampling (0-1, lower = more focused)
  --frequency-penalty <float>      Penalize repeated tokens (cannot combine with --presence-penalty)
  --presence-penalty <float>       Penalize tokens already used (cannot combine with --frequency-penalty)

\x1b[1mOUTPUT OPTIONS:\x1b[0m
  --language <code>                Preferred response language (ISO 639-1, e.g. "en", "fr")
  --images                         Include image results
  --related                        Include related follow-up questions
  --strip-thinking                 Remove <think> tags (research/reason)
  --model <model>                  Override the default model
  --json                           Output raw JSON response
  -h, --help                       Show this help

\x1b[1mPROMPTING TIPS:\x1b[0m
  - Always include today's date in the query for temporal context:
      "As of 2026-06-13, what is the latest..."
    Run \`date +%Y-%m-%d\` first if unsure. A date is also auto-injected as
    a safety net, but explicit in the query grounds the search better.
  - Be specific — the query seeds the web search directly.
  - Cap lists: "top 5" not "as many as possible".
  - Use --domain for site restrictions (prose like "only search X" is ignored).

\x1b[1mENVIRONMENT:\x1b[0m
  PERPLEXITY_BASE_URL    API base URL (default: http://localhost:3030/v1)
  PERPLEXITY_API_KEY     API key
  PERPLEXITY_TIMEOUT_MS  Timeout in ms (default: 300000)

\x1b[1mEXAMPLES:\x1b[0m
  perplexity search "As of 2026-06-13, latest bun release"
  perplexity search "breaking news" --recency hour
  perplexity search "React hooks" --domain "reactjs.org,developer.mozilla.org"
  perplexity search "quantum error correction" --mode academic --context-size high
  perplexity ask "As of 2026-06-13, explain quantum computing" --language ja
  perplexity research "state of AI agents as of 2026-06" --context-size high
  perplexity reason "Rust or Go for a CLI tool?" --temperature 0.2
`;

function parseArgs(argv: string[]) {
  const args = argv.slice(2); // skip bun/node and script path

  if (args.length === 0 || args.includes("-h") || args.includes("--help")) {
    console.log(HELP);
    process.exit(0);
  }

  const command = args[0];
  const validCommands = ["search", "ask", "research", "reason"];
  if (!validCommands.includes(command)) {
    console.error(`\x1b[31mError:\x1b[0m Unknown command "${command}"\n`);
    console.error(`Valid commands: ${validCommands.join(", ")}`);
    console.error(`Run "perplexity --help" for usage.`);
    process.exit(1);
  }

  let query = "";
  let recency: string | undefined;
  let domains: string[] | undefined;
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

  const queryParts: string[] = [];

  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--recency" && i + 1 < args.length) {
      recency = args[++i];
      if (!RECENCY_VALUES.includes(recency as any)) {
        console.error(`\x1b[31mError:\x1b[0m Invalid recency value "${recency}". Use: ${RECENCY_VALUES.join(", ")}`);
        process.exit(1);
      }
    } else if (arg === "--domain" && i + 1 < args.length) {
      domains = args[++i].split(",").map((d) => d.trim()).filter(Boolean);
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
      if (isNaN(temperature)) {
        console.error(`\x1b[31mError:\x1b[0m --temperature must be a number`);
        process.exit(1);
      }
    } else if (arg === "--max-tokens" && i + 1 < args.length) {
      maxTokens = parseInt(args[++i], 10);
      if (isNaN(maxTokens)) {
        console.error(`\x1b[31mError:\x1b[0m --max-tokens must be an integer`);
        process.exit(1);
      }
    } else if (arg === "--top-p" && i + 1 < args.length) {
      topP = parseFloat(args[++i]);
      if (isNaN(topP)) {
        console.error(`\x1b[31mError:\x1b[0m --top-p must be a number`);
        process.exit(1);
      }
    } else if (arg === "--frequency-penalty" && i + 1 < args.length) {
      frequencyPenalty = parseFloat(args[++i]);
      if (isNaN(frequencyPenalty)) {
        console.error(`\x1b[31mError:\x1b[0m --frequency-penalty must be a number`);
        process.exit(1);
      }
    } else if (arg === "--presence-penalty" && i + 1 < args.length) {
      presencePenalty = parseFloat(args[++i]);
      if (isNaN(presencePenalty)) {
        console.error(`\x1b[31mError:\x1b[0m --presence-penalty must be a number`);
        process.exit(1);
      }
    } else if (arg === "--language" && i + 1 < args.length) {
      languagePreference = args[++i];
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
    } else if (arg.startsWith("-")) {
      console.error(`\x1b[31mError:\x1b[0m Unknown option "${arg}"`);
      process.exit(1);
    } else {
      queryParts.push(arg);
    }
  }

  query = queryParts.join(" ");

  if (!query) {
    if (!process.stdin.isTTY) {
      // Will be handled below
    } else {
      console.error(`\x1b[31mError:\x1b[0m No query provided.`);
      console.error(`Usage: perplexity ${command} "your query here"`);
      process.exit(1);
    }
  }

  return {
    command, query, recency, domains, contextSize, searchMode, searchType,
    searchLangs, temperature, maxTokens, topP,
    frequencyPenalty, presencePenalty, languagePreference,
    returnImages, returnRelatedQuestions, stripThinking, model, rawJson,
  };
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf-8").trim();
}

async function main() {
  const {
    command, query: parsedQuery, recency, domains, contextSize,
    searchMode, searchType, searchLangs,
    temperature, maxTokens, topP, frequencyPenalty, presencePenalty,
    languagePreference, returnImages, returnRelatedQuestions,
    stripThinking, model, rawJson,
  } = parseArgs(process.argv);

  let query = parsedQuery;
  if (!query && !process.stdin.isTTY) {
    query = await readStdin();
  }

  if (!query) {
    console.error(`\x1b[31mError:\x1b[0m No query provided.`);
    process.exit(1);
  }

  const messages: Message[] = [{ role: "user", content: query }];

  const modelMap: Record<string, string> = {
    search: "sonar",
    ask: "sonar-pro",
    research: "sonar-deep-research",
    reason: "sonar-pro",
  };

  const systemExtraMap: Record<string, string | undefined> = {
    search: undefined,
    ask: undefined,
    research: undefined,
    reason: "Think step-by-step. Evaluate all angles before concluding. Structure your response with clear sections.",
  };

  try {
    const result = await performChatCompletion({
      messages,
      model: model ?? modelMap[command],
      stripThinking,
      systemExtra: systemExtraMap[command],
      recencyFilter: recency,
      searchDomainFilter: domains,
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
    });

    console.log(result);
  } catch (error) {
    console.error(`\x1b[31mError:\x1b[0m ${error instanceof Error ? error.message : error}`);
    process.exit(1);
  }
}

main();
