// synthesize.go reads candidate sessions and produces the recall block
// that gets injected back to Claude. Lossless content is the goal — pass
// the local LLM the actual prompt-and-response excerpts, NOT pre-computed
// summaries. The LLM picks what matters for the current query.
package recall

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/bughatti/void-memory/internal/indexer"
	"github.com/bughatti/void-memory/internal/ollama"
	"github.com/bughatti/void-memory/pkg/types"
)

// Synthesizer reads candidate sessions and produces a focused recall block.
type Synthesizer struct {
	llm   *ollama.Client
	model string
}

func NewSynthesizer(llm *ollama.Client, model string) *Synthesizer {
	return &Synthesizer{llm: llm, model: model}
}

// Synthesize takes the original user query, the routing result, and the
// scoped candidate sessions; loads recent excerpts from each candidate,
// and asks the LLM to produce a recall block tuned to the query.
func (s *Synthesizer) Synthesize(
	ctx context.Context,
	query string,
	route types.RouteResult,
	candidates []types.SessionMeta,
	maxTokens int,
) (*types.RecallResult, error) {
	if maxTokens <= 0 {
		maxTokens = 1500
	}
	if len(candidates) == 0 {
		return &types.RecallResult{
			Synthesis:     "",
			SessionsUsed:  nil,
			RoutingResult: route,
			GeneratedAt:   time.Now().UTC(),
		}, nil
	}

	excerpts, used := s.buildExcerpts(candidates, route.Entities, route.Keywords)
	if excerpts == "" {
		return &types.RecallResult{
			Synthesis:     "",
			SessionsUsed:  nil,
			RoutingResult: route,
			GeneratedAt:   time.Now().UTC(),
		}, nil
	}

	sys := buildSynthesisSystemPrompt(maxTokens)
	user := fmt.Sprintf(
		"CURRENT USER PROMPT:\n%s\n\nCANDIDATE PRIOR EXCERPTS:\n%s\n\nProduce the recall block now.",
		strings.TrimSpace(query),
		excerpts,
	)
	out, err := s.llm.Generate(ctx, ollama.GenerateRequest{
		Model:  s.model,
		System: sys,
		Prompt: user,
		Options: map[string]interface{}{
			"temperature": 0.2,
			// Approximate token cap; Ollama uses num_predict.
			"num_predict": maxTokens,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("synthesize llm: %w", err)
	}
	synthesis := strings.TrimSpace(out)
	return &types.RecallResult{
		Synthesis:     synthesis,
		SessionsUsed:  used,
		TokenEstimate: len(synthesis) / 4, // rough heuristic, 4 chars/token
		RoutingResult: route,
		GeneratedAt:   time.Now().UTC(),
	}, nil
}

// buildExcerpts opens each candidate's JSONL and extracts the prompts +
// short assistant snippets that mention any of the entities/keywords from
// the routing result. The output is what the synthesizer LLM reads.
//
// Cap total excerpt size at ~16k chars — that's ~4k tokens for Qwen,
// enough to give the LLM real content without blowing its context window.
func (s *Synthesizer) buildExcerpts(
	candidates []types.SessionMeta,
	entities []string,
	keywords []string,
) (string, []string) {
	const maxTotal = 16000
	const perSessionCap = 4000

	needles := make([]string, 0, len(entities)+len(keywords))
	for _, e := range entities {
		needles = append(needles, strings.ToLower(e))
	}
	for _, k := range keywords {
		needles = append(needles, strings.ToLower(k))
	}

	var sb strings.Builder
	used := make([]string, 0, len(candidates))
	for _, c := range candidates {
		if sb.Len() >= maxTotal {
			break
		}
		ps, err := indexer.ParseFile(c.JSONLPath)
		if err != nil {
			continue
		}
		excerpt := extractRelevantTurns(ps, needles, perSessionCap)
		if excerpt == "" {
			continue
		}
		sb.WriteString(fmt.Sprintf(
			"\n--- SESSION %s (%s, category=%s) ---\n",
			c.SessionID,
			c.EndTime.Format("2006-01-02"),
			c.Category,
		))
		sb.WriteString(excerpt)
		sb.WriteString("\n")
		used = append(used, c.SessionID)
	}
	return sb.String(), used
}

// extractRelevantTurns walks the parsed session, picks user prompts and the
// immediately-following assistant text turn whenever any needle appears in
// the user prompt. This gives the LLM the question + response pair as the
// unit of relevance — the actual decisions and their context.
func extractRelevantTurns(ps *indexer.ParsedSession, needles []string, cap int) string {
	if len(needles) == 0 {
		// No entity signal — return the start of the session as a fallback
		// (user prompts ordered by time).
		var sb strings.Builder
		for _, m := range ps.UserPrompts() {
			if sb.Len() >= cap {
				break
			}
			sb.WriteString("USER: ")
			sb.WriteString(truncate(m.Content, 500))
			sb.WriteString("\n")
		}
		return sb.String()
	}

	var sb strings.Builder
	prompts := ps.Messages
	for i, m := range prompts {
		if sb.Len() >= cap {
			break
		}
		if m.Type != "user" || m.IsToolResult {
			continue
		}
		content := strings.TrimSpace(m.Content)
		if content == "" {
			continue
		}
		// Skip noise we already filter elsewhere.
		if strings.HasPrefix(content, "<system-reminder>") ||
			strings.HasPrefix(content, "<local-command-stdout>") ||
			strings.HasPrefix(content, "<command-") ||
			strings.HasPrefix(content, "<task-notification>") {
			continue
		}
		lower := strings.ToLower(content)
		matched := false
		for _, n := range needles {
			if strings.Contains(lower, n) {
				matched = true
				break
			}
		}
		if !matched {
			continue
		}
		sb.WriteString("USER: ")
		sb.WriteString(truncate(content, 800))
		sb.WriteString("\n")
		// Pair with the next assistant text turn if available.
		for j := i + 1; j < len(prompts) && j < i+4; j++ {
			next := prompts[j]
			if next.Type != "assistant" || next.IsToolUse {
				continue
			}
			ac := strings.TrimSpace(next.Content)
			if ac == "" {
				continue
			}
			sb.WriteString("CLAUDE: ")
			sb.WriteString(truncate(ac, 1200))
			sb.WriteString("\n")
			break
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func buildSynthesisSystemPrompt(maxTokens int) string {
	return fmt.Sprintf(`You are the recall synthesizer for void-memory. Your job is to read PRIOR CONVERSATION EXCERPTS and produce a focused "<prior-work>" block that gives the assistant the context it needs to answer the CURRENT USER PROMPT well — without re-explaining unrelated history.

HARD RULES (violations are bugs, not stylistic preferences):

1. **No hallucination, ever.** Every fact in your output MUST appear in the SOURCE EXCERPTS provided below. Do NOT inject general knowledge about the user's domain (WoW levels, default versions, common configurations, "industry standard" advice). If something is not in the excerpts, do not include it. When in doubt, omit.

2. **Cite or omit.** Every concrete claim (number, name, file path, decision) must be tied to a session by citing the session_id in brackets like [SESSION_ID]. If you cannot cite, do not state.

3. **Prefer the most recent.** Excerpts are tagged with date headers like "--- SESSION xxx (2026-06-09, category=gameplay) ---". When the excerpts contain conflicting or evolving data (e.g., gear from 3 weeks ago vs gear from yesterday), STATE the most recent values and explicitly note "(superseded earlier value: X from <older date>)" only if the user might still care about the older value. Older transient facts (in-flight bugs, debugging context that was resolved) should be DROPPED, not summarized.

4. **Stale data warning.** If the most recent relevant excerpt is more than 7 days old, prepend a one-line "STALENESS: latest data is N days old, may need refresh" warning before the main block.

5. **Empty-is-OK.** If the excerpts contain nothing actually relevant to the current prompt, return an empty string. Returning nothing is correct behavior — false positives are worse than gaps.

6. **Verbatim values.** Preserve the user's actual numbers, item names, function names, file paths, and command outputs. Do not paraphrase or "clean up" their phrasing.

OUTPUT FORMAT:
- Output ONLY the recall block. No greetings, no "here is the context", no markdown fences.
- Wrap in tags: <prior-work>...</prior-work>
- Keep under approximately %d tokens (about %d characters).
- Order: load-bearing facts first (current state, key decisions, active blockers), secondary detail trailing.

Be terse, be precise, be faithful to the source excerpts.`, maxTokens, maxTokens*4)
}
