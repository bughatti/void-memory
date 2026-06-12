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
	synthesis := strings.TrimSpace(out)
	return &types.RecallResult{
		Synthesis:     synthesis,
		SessionsUsed:  used,
		TokenEstimate: len(synthesis) / 4,
		GeneratedAt:   time.Now().UTC(),
	}, nil
}
