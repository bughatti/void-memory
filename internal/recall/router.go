// Package recall handles the query-time path: classify the user's prompt,
// scope the catalog, synthesize a recall block from the top-N candidate
// sessions. The router (this file) is the classification + scoping step;
// the synthesizer (synthesize.go) is the LLM read-and-condense step.
package recall

import (
	"context"
	"fmt"
	"strings"

	"github.com/bughatti/void-memory/internal/catalog"
	"github.com/bughatti/void-memory/internal/ollama"
	"github.com/bughatti/void-memory/internal/taxonomy"
	"github.com/bughatti/void-memory/pkg/types"
)

// Router asks the local LLM to classify a user prompt into the taxonomy.
type Router struct {
	llm   *ollama.Client
	model string
}

func NewRouter(llm *ollama.Client, model string) *Router {
	return &Router{llm: llm, model: model}
}

// Route classifies the prompt and returns the routing result. The
// NeedsContext flag is set to false when the LLM judges that no prior
// context would help (e.g., "what's 2+2"); when false the caller should
// short-circuit retrieval entirely to save Claude tokens.
func (r *Router) Route(ctx context.Context, query string, recentEntities []string) (types.RouteResult, error) {
	sys := buildRouterSystemPrompt(recentEntities)
	user := fmt.Sprintf(
		"User prompt:\n%s\n\nRespond with ONLY the JSON object.",
		strings.TrimSpace(query),
	)
	var raw struct {
		NeedsContext bool     `json:"needs_context"`
		Category     string   `json:"category"`
		Entities     []string `json:"entities"`
		Keywords     []string `json:"keywords"`
		Reasoning    string   `json:"reasoning"`
	}
	if err := r.llm.GenerateJSON(ctx, r.model, sys, user, &raw); err != nil {
		return types.RouteResult{}, fmt.Errorf("route llm: %w", err)
	}
	return types.RouteResult{
		NeedsContext: raw.NeedsContext,
		Category:     normalizeCategory(raw.Category),
		Entities:     raw.Entities,
		Keywords:     raw.Keywords,
		Reasoning:    raw.Reasoning,
	}, nil
}

func buildRouterSystemPrompt(recentEntities []string) string {
	var sb strings.Builder
	sb.WriteString(`You are the routing classifier for void-memory. Given ONE user prompt from a Claude Code conversation, decide:

1) Would loading prior context from past sessions help answer it? (needs_context: true/false)
   - true: the prompt references a specific entity, project, file, or topic that the user has likely worked on before
   - false: standalone or generic ("what's 2+2", "list files in cwd", "explain async/await")

2) If true, classify into ONE category and extract relevant entities + keywords for scoped retrieval.

CATEGORIES (pick exactly one, must be one of these IDs):
`)
	for _, d := range taxonomy.Definitions {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", d.ID, d.Description))
	}
	sb.WriteString("\nEDGE-CASE RULES:")
	sb.WriteString(taxonomy.EdgeCaseRules)
	sb.WriteString(`
KNOWN ENTITIES (match where possible — list these by their canonical name, NOT a paraphrase):
`)
	sb.WriteString(strings.Join(taxonomy.AllSeedEntities(), ", "))
	if len(recentEntities) > 0 {
		sb.WriteString("\n\nENTITIES MENTIONED IN RECENT TURNS (these are top candidates for the current prompt):\n")
		sb.WriteString(strings.Join(recentEntities, ", "))
	}
	sb.WriteString(`

OUTPUT — return JSON matching this shape:
{
  "needs_context": true,
  "category": "<category-id>",
  "entities": ["<entity-canonical-name>", ...],
  "keywords": ["<verbatim-keyword>", ...],
  "reasoning": "<one short sentence on why this routing>"
}

When in doubt about needs_context, default to true and let downstream scoring filter — false-negatives are worse than false-positives here.`)
	return sb.String()
}

func normalizeCategory(raw string) types.Category {
	raw = strings.ToLower(strings.TrimSpace(raw))
	for _, d := range taxonomy.Definitions {
		if strings.ToLower(string(d.ID)) == raw {
			return d.ID
		}
	}
	switch raw {
	case "addon", "wow-addon":
		return types.CategoryAddons
	case "api", "blizzard-api":
		return types.CategoryWowAPI
	case "backend":
		return types.CategoryVoidScoutBackend
	case "play", "wow-play":
		return types.CategoryGameplay
	case "infrastructure", "devops":
		return types.CategoryInfra
	case "tools", "meta":
		return types.CategoryTooling
	}
	return types.CategoryAddons
}

// ScopeCandidates ranks catalog sessions by relevance to the routing result.
// Returns at most maxCandidates session metas. The scoring is intentionally
// simple — exact category match is a hard prefilter, entity overlap is the
// primary signal, recency is a small tiebreaker.
func ScopeCandidates(cat *catalog.Catalog, route types.RouteResult, maxCandidates int) []types.SessionMeta {
	// Try category-scoped first; if it has too few entity hits, widen.
	primary := cat.FindByCategory(string(route.Category))
	scored := scoreSessions(primary, route.Entities)
	if len(scored) >= maxCandidates {
		if len(scored) > maxCandidates {
			scored = scored[:maxCandidates]
		}
		return scored
	}

	// Widen to all categories — the user's prompt may have been
	// misclassified, and we'd rather over-recall than miss the actual match.
	all := cat.All()
	wide := scoreSessions(all, route.Entities)
	// Combine, dedupe, cap.
	seen := make(map[string]struct{}, len(scored))
	merged := make([]types.SessionMeta, 0, maxCandidates)
	for _, m := range scored {
		if _, ok := seen[m.SessionID]; !ok {
			merged = append(merged, m)
			seen[m.SessionID] = struct{}{}
		}
		if len(merged) >= maxCandidates {
			return merged
		}
	}
	for _, m := range wide {
		if _, ok := seen[m.SessionID]; !ok {
			merged = append(merged, m)
			seen[m.SessionID] = struct{}{}
		}
		if len(merged) >= maxCandidates {
			return merged
		}
	}
	return merged
}

// scoreSessions ranks sessions by entity-overlap count descending,
// breaking ties by recency.
func scoreSessions(sessions []types.SessionMeta, entities []string) []types.SessionMeta {
	if len(entities) == 0 {
		// No entity signal — sort by recency only.
		out := make([]types.SessionMeta, len(sessions))
		copy(out, sessions)
		for i := 1; i < len(out); i++ {
			for j := i; j > 0 && out[j].EndTime.After(out[j-1].EndTime); j-- {
				out[j], out[j-1] = out[j-1], out[j]
			}
		}
		return out
	}
	want := make(map[string]struct{}, len(entities))
	for _, e := range entities {
		want[strings.ToLower(e)] = struct{}{}
	}
	type scored struct {
		m    types.SessionMeta
		hits int
	}
	var ss []scored
	for _, m := range sessions {
		hits := 0
		for _, e := range m.Entities {
			if _, ok := want[strings.ToLower(e)]; ok {
				hits++
			}
		}
		if hits > 0 {
			ss = append(ss, scored{m: m, hits: hits})
		}
	}
	for i := 1; i < len(ss); i++ {
		for j := i; j > 0; j-- {
			a, b := ss[j-1], ss[j]
			if b.hits > a.hits || (b.hits == a.hits && b.m.EndTime.After(a.m.EndTime)) {
				ss[j-1], ss[j] = ss[j], ss[j-1]
			} else {
				break
			}
		}
	}
	out := make([]types.SessionMeta, len(ss))
	for i, s := range ss {
		out[i] = s.m
	}
	return out
}
