#!/usr/bin/env bun
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

// ============================================================================
// Configuration — override via env vars
// ============================================================================

const BASE_URL = process.env.PERPLEXITY_BASE_URL ?? "http://localhost:3030/v1";
const TIMEOUT_MS = Number(process.env.PERPLEXITY_TIMEOUT_MS ?? 300_000);
const API_KEY = process.env.PERPLEXITY_API_KEY ?? "";

const QUERY_POLICY =
  "\n\nInclude today's date in the query when temporal context matters (e.g. \"as of 2026-06-13, what is...\"). This grounds the search in current time. The tool also auto-injects date context as a safety net, but explicit is better.";

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
  enableSearchClassifier?: boolean;
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
  // Search params
  if (recencyFilter && RECENCY_VALUES.includes(recencyFilter as (typeof RECENCY_VALUES)[number])) {
    body.search_recency_filter = recencyFilter;
  }
  if (searchDomainFilter?.length) body.search_domain_filter = searchDomainFilter;
  if (searchContextSize) body.web_search_options = { search_context_size: searchContextSize };
  if (searchMode) body.search_mode = searchMode;
  if (searchType) body.search_type = searchType;
  if (searchLanguageFilter?.length) body.search_language_filter = searchLanguageFilter;
  if (enableSearchClassifier !== undefined) body.enable_search_classifier = enableSearchClassifier;
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

// Shared input schema fragments
const SEARCH_FILTER_PROPS = {
  search_domain_filter: {
    type: "array",
    items: { type: "string" },
    description:
      'Restrict search to specific domains. Prefix with "-" to exclude. Example: ["wikipedia.org", "-pinterest.com"]. Use this param for site restrictions — prose like "only search X" is ignored by the search backend.',
  },
  search_context_size: {
    type: "string",
    enum: ["low", "medium", "high"],
    description: 'Search depth vs cost. "high" = exhaustive (more sources, slower, costlier). "low" = fast/cheap. Omit for default (medium).',
  },
  search_mode: {
    type: "string",
    enum: ["web", "academic", "sec"],
    description: 'Search source mode. "web" = general web (default). "academic" = scholarly/research papers. "sec" = SEC filings.',
  },
  search_type: {
    type: "string",
    enum: ["fast", "pro", "auto"],
    description: 'Search type. "fast" = quicker/cheaper. "pro" = deeper/more thorough. "auto" = model decides.',
  },
  search_language_filter: {
    type: "array",
    items: { type: "string" },
    description: 'Filter search results by language. ISO 639-1 codes, up to 10. Example: ["en", "fr"].',
  },

} as const;

const GENERATION_PROPS = {
  temperature: {
    type: "number",
    description: "Sampling temperature (0 = deterministic, 2 = max randomness). Omit for model default.",
  },
  max_tokens: {
    type: "integer",
    description: "Maximum tokens in the response. Omit for no limit.",
  },
  top_p: {
    type: "number",
    description: "Nucleus sampling (0-1). Lower = more focused. Omit for default.",
  },
  frequency_penalty: {
    type: "number",
    description: "Penalize repeated tokens (-2 to 2). Cannot be used together with presence_penalty.",
  },
  presence_penalty: {
    type: "number",
    description: "Penalize tokens already present (-2 to 2). Cannot be used together with frequency_penalty.",
  },
  language_preference: {
    type: "string",
    description: 'Preferred response language as ISO 639-1 code (e.g. "en", "fr", "ja"). Omit for auto-detect.',
  },
  return_images: {
    type: "boolean",
    description: "Set to true to include image results in the response.",
  },
  return_related_questions: {
    type: "boolean",
    description: "Set to true to include related follow-up questions in the response.",
  },
} as const;

const MESSAGES_PROP = {
  type: "array",
  description: "Conversation history. Provide either this or 'query'.",
  items: {
    type: "object",
    properties: {
      role: { type: "string", description: "system, user, or assistant" },
      content: { type: "string", description: "Message content" },
    },
    required: ["role", "content"],
  },
} as const;

const RESPONSE_FORMAT_PROP = {
  type: "object",
  description:
    'Enforce structured output. Must use json_schema format: { "type": "json_schema", "json_schema": { "name": "my_name", "schema": { "type": "object", "properties": {...}, "required": [...] } } }. "json_object" is NOT supported.',
} as const;

const TOOLS = [
  {
    name: "perplexity_search",
    description:
      `Web search with citations (sonar). Best for current events, facts, documentation, quick lookups.

Prompting rules:
- Query IS the search seed — be specific (include entities, versions, constraints)
- Cap lists explicitly ("top 5") — vague requests scatter results
- Don't ask for URLs in query — citations are returned separately
- Use search_domain_filter for site restrictions (prose "only search X" is IGNORED)
- search_context_size "high" = deeper research, "low" = faster/cheaper` + QUERY_POLICY,
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "The search query — be specific (entities, versions, constraints). This seeds the web search." },
        recency_filter: {
          type: "string",
          enum: ["hour", "day", "week", "month"],
          description: "Limit results to recent timeframe. Omit for no restriction.",
        },
        ...SEARCH_FILTER_PROPS,
        ...GENERATION_PROPS,
      },
      required: ["query"],
    },
  },
  {
    name: "perplexity_ask",
    description:
      `Multi-turn conversation with web-grounded responses (sonar-pro). Use when prior context/messages are needed, or for follow-up questions.

Prompting rules:
- Query IS the search seed — be specific (include entities, versions, constraints)
- Cap lists explicitly ("top 5")
- Use search_domain_filter for site restrictions (prose is IGNORED by search backend)
- Use response_format for machine-readable JSON output (requires json_schema, NOT json_object)
- Don't ask for URLs in query — citations are returned separately` + QUERY_POLICY,
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Simple one-shot question (shorthand for a single user message)",
        },
        messages: MESSAGES_PROP,
        ...SEARCH_FILTER_PROPS,
        response_format: RESPONSE_FORMAT_PROP,
        ...GENERATION_PROPS,
      },
    },
  },
  {
    name: "perplexity_research",
    description:
      `Deep research across hundreds of sources (sonar-deep-research). Produces thorough report-style output. May take 30-60s+.

Prompting rules:
- Frame as a research task — specify dimensions to cover (e.g. "focus on: cost, safety, scalability")
- Be specific in the query — it seeds multi-step search; vague = scattered
- Cap scope: "top 5 approaches" or "last 2 years" prevents unfocused sprawl
- Use search_domain_filter for source restrictions (prose is IGNORED)
- search_context_size "high" recommended for thorough research
- Don't ask for URLs in query — citations are returned separately
- Allow uncertainty: "say if info unavailable" reduces hallucination` + QUERY_POLICY,
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Research question — be specific, state dimensions to cover, cap scope",
        },
        messages: MESSAGES_PROP,
        strip_thinking: {
          type: "boolean",
          description: "Remove <think> tags from response to save context tokens.",
          default: false,
        },
        ...SEARCH_FILTER_PROPS,
        ...GENERATION_PROPS,
      },
    },
  },
  {
    name: "perplexity_reason",
    description:
      `Analytical reasoning with web grounding (sonar-pro). Use for evaluating options, pros/cons, logical problem-solving, decision-making.

Prompting rules:
- Frame as a decision/analysis: "evaluate X vs Y", "what are the tradeoffs of..."
- Be specific — include constraints, context, and what matters most
- Cap output: "give 3 options with pros/cons" prevents sprawl
- Use response_format for structured JSON output (requires json_schema, NOT json_object)
- Use search_domain_filter for source restrictions (prose is IGNORED)
- Don't ask for URLs in query — citations are returned separately` + QUERY_POLICY,
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Question to reason about (shorthand for a single user message)",
        },
        messages: MESSAGES_PROP,
        strip_thinking: {
          type: "boolean",
          description: "Remove <think> tags from response to save context tokens.",
          default: false,
        },
        ...SEARCH_FILTER_PROPS,
        response_format: RESPONSE_FORMAT_PROP,
        ...GENERATION_PROPS,
      },
    },
  },
] as const;

// ============================================================================
// Server
// ============================================================================

const server = new Server(
  { name: "perplexity-mcp", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const a = (args ?? {}) as Record<string, unknown>;

  // Extract shared optional params
  function sharedOpts(): Partial<CompletionOptions> {
    return {
      // Search
      searchDomainFilter: Array.isArray(a.search_domain_filter) ? a.search_domain_filter as string[] : undefined,
      searchContextSize: a.search_context_size as string | undefined,
      searchMode: a.search_mode as string | undefined,
      searchType: a.search_type as string | undefined,
      searchLanguageFilter: Array.isArray(a.search_language_filter) ? a.search_language_filter as string[] : undefined,
      enableSearchClassifier: typeof a.enable_search_classifier === "boolean" ? a.enable_search_classifier : undefined,

      // Generation
      responseFormat: a.response_format as Record<string, unknown> | undefined,
      temperature: typeof a.temperature === "number" ? a.temperature : undefined,
      maxTokens: typeof a.max_tokens === "number" ? a.max_tokens : undefined,
      topP: typeof a.top_p === "number" ? a.top_p : undefined,
      frequencyPenalty: typeof a.frequency_penalty === "number" ? a.frequency_penalty : undefined,
      presencePenalty: typeof a.presence_penalty === "number" ? a.presence_penalty : undefined,
      // Response enrichment
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
      result = await performChatCompletion({
        messages: [{ role: "user", content: a.query as string }],
        model: "sonar",
        recencyFilter: a.recency_filter as string | undefined,
        ...sharedOpts(),
      });
    } else if (name === "perplexity_ask") {
      result = await performChatCompletion({
        messages: resolveMessages(),
        model: "sonar-pro",
        ...sharedOpts(),
      });
    } else if (name === "perplexity_research") {
      result = await performChatCompletion({
        messages: resolveMessages(),
        model: "sonar-deep-research",
        stripThinking: (a.strip_thinking as boolean) ?? false,
        ...sharedOpts(),
      });
    } else if (name === "perplexity_reason") {
      result = await performChatCompletion({
        messages: resolveMessages(),
        model: "sonar-pro",
        stripThinking: (a.strip_thinking as boolean) ?? false,
        systemExtra:
          "Think step-by-step. Evaluate all angles before concluding. Structure your response with clear sections.",
        ...sharedOpts(),
      });
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
