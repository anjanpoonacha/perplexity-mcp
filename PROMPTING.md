# Perplexity Sonar API — Prompting Best Practices

Consolidated from official docs ([Sonar Prompt Guide](https://docs.perplexity.ai/docs/sonar/prompt-guide), [Agent API Prompt Guide](https://docs.perplexity.ai/docs/agent-api/prompt-guide)) and deep research.

---

## Core Principles

1. **User message = search seed** — specificity here directly improves retrieval quality
2. **System prompt = behavioral contract** — role, tone, formatting, grounding rules
3. **Parameters > prose** — use API parameters for hard constraints; prose instructions like "only search Wikipedia" are ignored by the search backend
4. **Never ask for URLs in text** — citations come back in the API response's `citations` array; asking in prose causes hallucination
5. **No few-shot content** — pasting a worked example answer hijacks the search toward the example's topic. Structure examples (JSON skeleton, bullet pattern) are fine
6. **Cap list lengths** — "top 5" not "as many as possible"
7. **Allow "I don't know"** — explicitly permit uncertainty to reduce confident fabrication

---

## System vs User Message

| System prompt | User message |
|---|---|
| Role, persona, tone | The actual query (search seed) |
| Formatting rules (Markdown, JSON, bullets) | Specific constraints for this query |
| Safety/grounding rules | Entities, versions, scope |
| Citation policy | Desired output format for this answer |
| Priority rules (what wins when constraints conflict) | — |

---

## API Parameters That Control Search (NOT prose)

| Parameter | Effect | Values |
|---|---|---|
| `search_recency_filter` | Time-window filter on results | `hour`, `day`, `week`, `month` |
| `search_domain_filter` | Include/exclude specific domains | Array of domains, prefix `-` to exclude |
| `web_search_options.search_context_size` | Depth vs cost tradeoff | `low` (cheap/shallow), `medium`, `high` (deep/expensive) |
| `response_format` | Enforce structured output | `{ type: "json_object" }` or JSON schema |
| `temperature` | Randomness (0 = deterministic) | 0.0–2.0 |
| `max_tokens` | Cap output length | integer |

**Critical:** Writing "search only on Wikipedia" or "only recent results" in the prompt text has NO effect on search retrieval. Use the parameters above.

---

## Model Selection

| Model | Best for | Latency | Prompt emphasis |
|---|---|---|---|
| `sonar` | Quick facts, current events, docs | Fast | Specificity, format |
| `sonar-pro` | Complex reasoning, multi-step Q&A | Medium | Format, conciseness, determinism |
| `sonar-deep-research` | Comprehensive reports, literature reviews | 30s–15min | Depth, citation, research methodology |

### sonar-pro tips
- Best for format-strict outputs (JSON, code, diffs)
- Explicitly forbid commentary when you need raw output: "Return only valid JSON, no surrounding text"
- For explanations, specify audience level and desired examples
- For code: include language, version, libraries, and whether you want comments

### sonar-deep-research tips
- Frame it as a research assistant, not a Q&A bot
- Instruct it to decompose complex questions into sub-questions
- Specify dimensions: "focus on: cost, safety, environmental impact"
- Stage calls: ask for outline first, then generate sections
- Control depth vs breadth explicitly
- Time horizons: "prioritize sources from last 2 years"

---

## Do vs Don't

| Do | Don't |
|---|---|
| Use `search_domain_filter` for site restrictions | Write "search only on MDN" in prose |
| Explicitly cap lists ("top 5") | Ask for "as many as possible" |
| Use `response_format` for JSON output | Paste full example answers to force structure |
| Be specific in the user query | Ask vague questions like "tell me about movies" |
| Ask model to say "info not available" if unsure | Let it guess when context is insufficient |
| Use `search_recency_filter` param for time constraints | Write "only recent articles" in prose |
| Keep system prompts concise and focused | Write multi-page system prompts (risk truncation) |
| Separate list items with blank lines | Cram everything into dense paragraphs |

---

## Anti-Hallucination Patterns

- Include: "If no relevant sources found or data are inconsistent, say so explicitly rather than guessing"
- Don't substitute related results — if search returns wrong company/year, model should say so
- Instruct model to "distinguish between well-established facts and speculative/contested claims"
- Never force URLs in prose — rely on API `citations` field

---

## Output Formatting

- Explicitly state format: paragraph, table, bullets, numbered list
- For machine-readable output, use `response_format` with JSON schema
- Separate list items with blank lines (Sonar's post-processor handles these better)
- Few-shot structure only (shape, not content)

---

## Context Management

- Keep system prompts concise — overly long prompts may trigger partial truncation
- For long conversations, summarize state rather than sending full raw history
- For large research, use hierarchical chunking: summarize chunks, then synthesize summaries
- Place most important instructions near the end of context (models weight recent tokens more)

---

## MCP Server & CLI: Supported Parameters

All parameters are now fully supported and pass through to the Perplexity API.

### Per-tool parameter availability

| Parameter | search | ask | research | reason |
|---|---|---|---|---|
| `search_domain_filter` | yes | yes | yes | yes |
| `search_context_size` | yes | yes | yes | yes |
| `recency_filter` | yes | — | — | — |
| `response_format` | — | yes | — | yes |
| `temperature` | yes | yes | yes | yes |
| `max_tokens` | yes | yes | yes | yes |
| `strip_thinking` | — | — | yes | yes |

### CLI flags (mirror MCP params)

```
--recency <hour|day|week|month>
--domain <comma-separated domains>      → search_domain_filter
--context-size <low|medium|high>        → web_search_options.search_context_size
--temperature <float>
--max-tokens <int>
--strip-thinking
```

### Implementation notes
- All parameters pass straight through to the request body
- `search_domain_filter` and `search_recency_filter` are search-layer params (affect retrieval)
- `response_format`, `temperature`, `max_tokens` are generation-layer params (affect output)
- `web_search_options` controls search depth/cost tradeoff
- `response_format` must be `{ "type": "json_schema", "json_schema": { "name": "...", "schema": {...} } }` — `json_object` type is NOT supported by Perplexity

---

## Practical Prompt Templates

### General research (sonar-deep-research)

**System:**
> You are a precise research assistant. Always answer in Markdown. Cite sources inline using bracketed indices. If search results are missing or off-topic, explicitly say so and do not guess.

**User:**
> Compare technology A vs B for use in [context]. Focus on: performance, cost, ecosystem maturity.
> Output: 2-3 sentence overview, then a Markdown table (Criterion | A | B | Notes), then a 1-paragraph recommendation.
> Limit to 5 most relevant trade-offs.

### Factual short answer (sonar / sonar-pro)

> Question: Is feature X supported in framework Y version Z?
> Output: Direct yes/no in first sentence. 3-5 bullets with reasoning and version details.
> If sources disagree or info unavailable, say that explicitly.

### Strict JSON (sonar-pro)

> From the following text, extract all entities. Return only valid JSON (no markdown, no explanation):
> ```json
> [{ "type": "person|org|date", "value": "...", "context": "..." }]
> ```

---

## References

- [Sonar Prompt Guide](https://docs.perplexity.ai/docs/sonar/prompt-guide)
- [Agent API Prompt Guide](https://docs.perplexity.ai/docs/agent-api/prompt-guide)
- [Sonar Features](https://docs.perplexity.ai/docs/sonar/features)
- [Promptfoo Perplexity Provider](https://www.promptfoo.dev/docs/providers/perplexity/)
