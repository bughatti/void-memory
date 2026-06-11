// extract.go runs the LLM-driven metadata extraction step: given a
// ParsedSession, ask Ollama to produce {category, sub_topics, entities,
// summary, tags}. The taxonomy package supplies category definitions and
// the edge-case rules; the LLM is constrained to JSON output.
package indexer

import (
	"context"
	"fmt"
	"strings"

	"github.com/bughatti/void-memory/internal/ollama"
	"github.com/bughatti/void-memory/internal/taxonomy"
	"github.com/bughatti/void-memory/pkg/types"
)

// Extractor builds a SessionMeta from a ParsedSession using the local LLM.
type Extractor struct {
	llm   *ollama.Client
	model string
}

func NewExtractor(llm *ollama.Client, model string) *Extractor {
	return &Extractor{llm: llm, model: model}
}

// Extract runs the classification prompt against the LLM. Returns a
// SessionMeta with everything but SessionID, JSONLPath, timestamps, and
// PromptCount filled — the caller (the indexer loop) populates those from
// the ParsedSession.
func (e *Extractor) Extract(ctx context.Context, ps *ParsedSession) (types.SessionMeta, error) {
	prompts := ps.UserPrompts()
	if len(prompts) == 0 {
		return types.SessionMeta{
			Category: types.CategoryTooling,
			Summary:  "(empty session, no user prompts)",
		}, nil
	}
	// Stratified sample across the session timeline so we capture topic
	// SHIFTS, not just the opening theme. For a 10-week session, the user's
	// focus changes — gear in week 1, addon work in week 3, backend in week
	// 8. Sampling only the front buckets the whole session by its earliest
	// topic. We pick samples evenly spaced across the full prompt list.
	const sampleSize = 120
	excerpts := make([]string, 0, sampleSize)
	step := 1
	if len(prompts) > sampleSize {
		step = len(prompts) / sampleSize
		if step < 1 {
			step = 1
		}
	}
	for i := 0; i < len(prompts); i += step {
		m := prompts[i]
		s := strings.ReplaceAll(m.Content, "\n", " ")
		if len(s) > 200 {
			s = s[:200] + "..."
		}
		ts := m.Timestamp.Format("2006-01-02")
		excerpts = append(excerpts, fmt.Sprintf("- [%s] %s", ts, s))
		if len(excerpts) >= sampleSize {
			break
		}
	}

	// Deterministic entity pre-scan across the FULL session — strings.Contains
	// over every user prompt's text. The LLM gets this confirmed list as a
	// hint, fixing the failure mode where stratified sampling misses entity
	// mentions that DID happen in unsampled windows.
	detectedEntities := scanEntities(prompts, taxonomy.AllSeedEntities())

	sys := buildExtractionSystemPrompt()
	detectedNote := ""
	if len(detectedEntities) > 0 {
		detectedNote = fmt.Sprintf(
			"\n\nKNOWN ENTITIES DETECTED IN THIS SESSION (via exact string match across all %d prompts — confirmed present, include any that are relevant):\n%s",
			len(prompts),
			strings.Join(detectedEntities, ", "),
		)
	}
	user := fmt.Sprintf(
		"Session ID: %s\nDate range: %s to %s\nUser prompts in this session (most recent last):\n%s%s\n\nClassify per the rules above. Respond with ONLY the JSON object — no prose, no markdown fences.",
		ps.SessionID,
		ps.StartTime.Format("2006-01-02"),
		ps.EndTime.Format("2006-01-02"),
		strings.Join(excerpts, "\n"),
		detectedNote,
	)

	var raw struct {
		Category  string   `json:"category"`
		SubTopics []string `json:"sub_topics"`
		Entities  []string `json:"entities"`
		Summary   string   `json:"summary"`
		Tags      []string `json:"tags"`
	}
	if err := e.llm.GenerateJSON(ctx, e.model, sys, user, &raw); err != nil {
		return types.SessionMeta{}, fmt.Errorf("extract llm: %w", err)
	}

	cat := normalizeCategory(raw.Category)
	// Union deterministic detection with LLM selection. The LLM is allowed
	// to ADD entities it spots in context (e.g., a new character name we
	// haven't seeded), but it must NOT drop known entities that actually
	// appeared in the session — that's how Vede was getting filtered when
	// the LLM judged it "not relevant" to the categorized topic.
	mergedEntities := dedupeStrings(append(detectedEntities, raw.Entities...))
	return types.SessionMeta{
		Category:  cat,
		SubTopics: dedupeStrings(raw.SubTopics),
		Entities:  mergedEntities,
		Summary:   strings.TrimSpace(raw.Summary),
		Tags:      dedupeStrings(raw.Tags),
	}, nil
}

func buildExtractionSystemPrompt() string {
	var sb strings.Builder
	sb.WriteString(`You are the indexing classifier for void-memory, a persistent-context tool for Claude Code conversations. Your job is to read a list of user prompts from one conversation session and produce structured metadata as strict JSON.

CATEGORIES (pick EXACTLY ONE — no other strings allowed):
`)
	for _, d := range taxonomy.Definitions {
		sb.WriteString(fmt.Sprintf("- %s: %s\n  sub-topics: %s\n", d.ID, d.Description, strings.Join(d.SubTopics, ", ")))
	}
	sb.WriteString("\nEDGE-CASE RULES:")
	sb.WriteString(taxonomy.EdgeCaseRules)
	sb.WriteString(`
KNOWN ENTITIES (match against these where applicable, but add new ones you observe):
`)
	sb.WriteString(strings.Join(taxonomy.AllSeedEntities(), ", "))
	sb.WriteString(`

CROSS-CUTTING TAGS (optional list, choose any that apply):
- decision-vs-execution: session contained analytical decisions (not just commands)
- error-encountered: a real bug or failure was diagnosed
- published-artifact: something shipped (commit, release, deploy)
- verified-vs-speculative: conclusions were source-grounded (not guessed)
- temporal-relevance-short: advice that decays fast (specific gear, in-flight work)
- temporal-relevance-long: research/decisions worth preserving for months

OUTPUT FORMAT — return JSON exactly matching this shape, no prose:
{
  "category": "<one of the category IDs above>",
  "sub_topics": ["<sub-topic-id>", ...],
  "entities": ["<entity-name>", ...],
  "summary": "<2-3 sentence plain-English summary of what was worked on>",
  "tags": ["<tag-name>", ...]
}

Be conservative with entities — only list things that actually came up in the prompts.`)
	return sb.String()
}

// normalizeCategory snaps the LLM's category string to a valid enum value.
// Defaults to addons if unknown (most common bucket from the research).
func normalizeCategory(raw string) types.Category {
	raw = strings.ToLower(strings.TrimSpace(raw))
	for _, d := range taxonomy.Definitions {
		if strings.ToLower(string(d.ID)) == raw {
			return d.ID
		}
	}
	// Loose-match common synonyms the LLM might use.
	switch raw {
	case "addon", "wow-addon", "wow-addons":
		return types.CategoryAddons
	case "wow", "api", "blizzard-api", "wow-research":
		return types.CategoryWowAPI
	case "backend", "voidscout":
		return types.CategoryVoidScoutBackend
	case "play", "wow-play":
		return types.CategoryGameplay
	case "infrastructure", "devops", "nas":
		return types.CategoryInfra
	case "tools", "tool", "meta":
		return types.CategoryTooling
	}
	return types.CategoryAddons
}

// scanEntities does a deterministic substring scan over all user prompts in
// the session, returning known seed entities that appear at least once.
// Case-insensitive, whole-word-ish (we lowercase both sides). Cheap — a few
// MB of text against ~100 needles takes <50ms.
func scanEntities(prompts []types.Message, seeds []string) []string {
	var bb strings.Builder
	for _, m := range prompts {
		bb.WriteString(m.Content)
		bb.WriteByte(' ')
	}
	hay := strings.ToLower(bb.String())
	seen := make(map[string]struct{})
	var out []string
	for _, s := range seeds {
		key := strings.ToLower(s)
		if _, dup := seen[key]; dup {
			continue
		}
		if strings.Contains(hay, key) {
			seen[key] = struct{}{}
			out = append(out, s)
		}
	}
	return out
}

func dedupeStrings(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		key := strings.ToLower(s)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, s)
	}
	return out
}
