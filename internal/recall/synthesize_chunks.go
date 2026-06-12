package recall

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/bughatti/void-memory/internal/ollama"
	"github.com/bughatti/void-memory/internal/retrieval"
	"github.com/bughatti/void-memory/pkg/types"
)

// Synthesizer reads retrieved chunks and produces the recall block returned to
// Claude. (The legacy whole-session router+synthesizer was removed once hybrid
// retrieval was validated; this is the only synthesis path now.)
type Synthesizer struct {
	llm   *ollama.Client
	model string
}

func NewSynthesizer(llm *ollama.Client, model string) *Synthesizer {
	return &Synthesizer{llm: llm, model: model}
}

// stripCodeFence trims a wrapping markdown code fence if the model added one
// despite the prompt forbidding it (small models sometimes do). Guarantees the
// recall block isn't polluted with ``` lines regardless of model behavior.
func stripCodeFence(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```") {
		if i := strings.IndexByte(s, '\n'); i >= 0 {
			s = s[i+1:]
		}
		if j := strings.LastIndex(s, "```"); j >= 0 {
			s = s[:j]
		}
	}
	return strings.TrimSpace(s)
}

func buildSynthesisSystemPrompt(maxTokens int) string {
	return fmt.Sprintf(`You are the recall synthesizer for void-memory. Your job is to read PRIOR CONVERSATION EXCERPTS and produce a focused recall block that gives the assistant the context it needs to answer the CURRENT USER PROMPT well — without re-explaining unrelated history.

HARD RULES (violations are bugs, not stylistic preferences):

1. **No hallucination, ever.** Every fact in your output MUST appear in the SOURCE EXCERPTS provided below. Do NOT inject general knowledge about the user's domain (WoW levels, default versions, common configurations, "industry standard" advice). If something is not in the excerpts, do not include it. When in doubt, omit.

2. **Cite or omit.** Every concrete claim (number, name, file path, decision) must be tied to a session by citing the session_id in brackets like [SESSION_ID]. If you cannot cite, do not state.

3. **Prefer the most recent.** Excerpts are tagged with session ids and chunk ids. When the excerpts contain conflicting or evolving data, STATE the most recent values and note "(superseded earlier value: X)" only if the user might still care about the older value. Resolved transient facts (in-flight bugs, debugging context) should be DROPPED, not summarized.

4. **Empty-is-OK.** If the excerpts contain nothing actually relevant to the current prompt, return an empty string. Returning nothing is correct — false positives are worse than gaps.

5. **Verbatim values.** Preserve the user's actual numbers, item names, function names, file paths, and command outputs. Do not paraphrase or "clean up" their phrasing.

OUTPUT FORMAT (follow EXACTLY):
- Output ONLY the recall block. No greetings, no "here is the context".
- Wrap the ENTIRE output in EXACTLY these tags: <prior-work> ... </prior-work>. Do NOT use any other tag name (not <recall_block>, not <priority>). Do NOT wrap it in markdown code fences (no triple backticks).
- Keep under approximately %d tokens (about %d characters).
- Order: load-bearing facts first (current state, key decisions, active blockers), secondary detail trailing.

Be terse, be precise, be faithful to the source excerpts.`, maxTokens, maxTokens*4)
}

// SynthesizeChunks is the hybrid-retrieval synthesis path: instead of searching
// whole candidate sessions, it is handed the already-retrieved top-K chunks and
// produces the recall block over just those. This is the "defer the LLM to query
// time over a small, precisely-retrieved set" step — the index did the work, the
// LLM only condenses. Reuses the same faithfulness system prompt (cite-or-omit,
// no hallucination) as the legacy synthesizer.
func (s *Synthesizer) SynthesizeChunks(
	ctx context.Context,
	query string,
	results []retrieval.Result,
	maxTokens int,
) (*types.RecallResult, error) {
	if maxTokens <= 0 {
		maxTokens = 1500
	}
	if len(results) == 0 {
		return &types.RecallResult{GeneratedAt: time.Now().UTC()}, nil
	}

	var sb strings.Builder
	used := make([]string, 0, len(results))
	seen := make(map[string]bool)
	for _, r := range results {
		// Label each excerpt with its session id so the synthesizer can cite
		// [SESSION_ID] per the faithfulness prompt's "cite or omit" rule.
		sb.WriteString(fmt.Sprintf("\n--- EXCERPT [%s] (chunk %s) ---\n", r.Chunk.SessionID, r.Chunk.ID))
		sb.WriteString(r.Chunk.Text)
		sb.WriteString("\n")
		if !seen[r.Chunk.SessionID] {
			seen[r.Chunk.SessionID] = true
			used = append(used, r.Chunk.SessionID)
		}
	}

	sys := buildSynthesisSystemPrompt(maxTokens)
	user := fmt.Sprintf(
		"CURRENT USER PROMPT:\n%s\n\nRETRIEVED PRIOR EXCERPTS:\n%s\n\nProduce the recall block now.",
		strings.TrimSpace(query),
		sb.String(),
	)
	out, err := s.llm.Generate(ctx, ollama.GenerateRequest{
		Model:  s.model,
		System: sys,
		Prompt: user,
		Options: map[string]interface{}{
			"temperature": 0.2,
			"num_predict": maxTokens,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("synthesize chunks llm: %w", err)
	}
	synthesis := stripCodeFence(out)
	return &types.RecallResult{
		Synthesis:     synthesis,
		SessionsUsed:  used,
		TokenEstimate: len(synthesis) / 4,
		GeneratedAt:   time.Now().UTC(),
	}, nil
}
