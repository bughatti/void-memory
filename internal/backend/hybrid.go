// hybrid.go is the LocalBackend's hybrid-retrieval path: ingestion (transcripts
// -> chunks -> embeddings -> HybridIndex on disk) and query (embed -> hybrid
// search -> synthesize over the top-K chunks). Gated by config so the legacy
// LLM-routing path stays the default until this is validated.
package backend

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	"github.com/bughatti/void-memory/internal/indexer"
	"github.com/bughatti/void-memory/internal/ollama"
	"github.com/bughatti/void-memory/internal/retrieval"
	"github.com/bughatti/void-memory/pkg/types"
)

// hybridIndexPath is where the local hybrid index is persisted.
func hybridIndexPath(dataDir string) string {
	return filepath.Join(dataDir, "hybrid-index.gob")
}

// BuildHybridIndex walks every top-level transcript under projectsDir, chunks
// each session, embeds every chunk with the small CPU embed model, and returns
// the populated index. Embedding is the slow part (one Ollama call per chunk),
// but it runs on CPU and never touches the synthesis GPU. Progress is logged.
func BuildHybridIndex(
	ctx context.Context,
	projectsDir string,
	emb *ollama.Client,
	embModel string,
	opts retrieval.ChunkOptions,
) (*retrieval.HybridIndex, error) {
	paths, err := indexer.EnumerateProjectsDir(projectsDir)
	if err != nil {
		return nil, fmt.Errorf("enumerate transcripts: %w", err)
	}
	h := retrieval.NewHybridIndex(embModel)
	embedded := 0
	for _, p := range paths {
		ps, err := indexer.ParseFile(p)
		if err != nil {
			fmt.Fprintf(stderrLog, "hybrid: skip %s: %v\n", p, err)
			continue
		}
		chunks := retrieval.ChunkMessages(ps.SessionID, ps.Messages, opts)
		for _, c := range chunks {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
			}
			// CPU-pinned: a batch reindex must never compete with a foreground
			// GPU app (e.g. WoW) for VRAM — on an 8 GB card that contention can
			// crash the display driver. Latency doesn't matter for a background
			// build, so CPU is the safe default.
			vec, err := emb.EmbeddingsCPU(ctx, embModel, c.Text)
			if err != nil {
				return nil, fmt.Errorf("embed chunk %s: %w", c.ID, err)
			}
			h.Add(c, vec)
			embedded++
			if embedded%100 == 0 {
				fmt.Fprintf(stderrLog, "hybrid: embedded %d chunks...\n", embedded)
			}
		}
	}
	fmt.Fprintf(stderrLog, "hybrid: indexed %d chunks from %d sessions (dim=%d)\n", h.Len(), len(paths), h.Dim)
	return h, nil
}

// ReindexHybrid rebuilds the hybrid index from scratch and persists it, then
// swaps it in for live queries.
func (b *LocalBackend) ReindexHybrid(ctx context.Context) error {
	if b.embModel == "" {
		return fmt.Errorf("no embedding model configured (set VOID_MEMORY_EMBED_MODEL, e.g. nomic-embed-text)")
	}
	h, err := BuildHybridIndex(ctx, b.projectsDir, b.llm, b.embModel, retrieval.ChunkOptions{})
	if err != nil {
		return err
	}
	if err := h.Save(hybridIndexPath(b.dataDir)); err != nil {
		return fmt.Errorf("save hybrid index: %w", err)
	}
	b.hybrid = h
	return nil
}

// hybridRecall answers a query via the hybrid index: embed the query, hybrid-
// search for the top-K chunks, and synthesize over just those. Falls back to
// lexical-only retrieval if the embedder is briefly unavailable.
func (b *LocalBackend) hybridRecall(ctx context.Context, query string, hints RecallHints) (*types.RecallResult, error) {
	const topK = 8

	// Query embed is a single cheap call — CPU-pin it so it never loads the
	// embedder onto the GPU. Only the synthesis LLM uses the GPU at recall time.
	var qvec []float32
	if v, err := b.llm.EmbeddingsCPU(ctx, b.embModel, query); err != nil {
		fmt.Fprintf(stderrLog, "hybrid: query embed failed, lexical-only: %v\n", err)
	} else {
		qvec = v
	}

	results := b.hybrid.Search(qvec, query, topK, retrieval.SearchOptions{})
	if len(results) == 0 {
		return &types.RecallResult{GeneratedAt: time.Now().UTC()}, nil
	}
	return b.syn.SynthesizeChunks(ctx, query, results, hints.MaxTokens)
}

// HybridSearch runs retrieval ONLY (no LLM synthesis) — for validation/debug.
// Query embedding is CPU-pinned so it never competes with a foreground GPU app;
// the whole call is GPU-free, demonstrating the core thesis.
func (b *LocalBackend) HybridSearch(ctx context.Context, query string, k int) ([]retrieval.Result, error) {
	if b.hybrid == nil || b.hybrid.Len() == 0 {
		return nil, fmt.Errorf("no hybrid index loaded; run `void-memory reindex` first")
	}
	var qv []float32
	if v, err := b.llm.EmbeddingsCPU(ctx, b.embModel, query); err != nil {
		fmt.Fprintf(stderrLog, "hybrid search: query embed failed, lexical-only: %v\n", err)
	} else {
		qv = v
	}
	return b.hybrid.Search(qv, query, k, retrieval.SearchOptions{}), nil
}

// loadHybridIndex attempts to load a persisted index at startup. A missing
// index is not an error — the backend serves legacy recall until ReindexHybrid
// is run.
func (b *LocalBackend) loadHybridIndex() {
	h, err := retrieval.LoadHybridIndex(hybridIndexPath(b.dataDir))
	if err != nil {
		fmt.Fprintf(stderrLog, "hybrid: no index loaded (%v); run `void-memory reindex`\n", err)
		return
	}
	b.hybrid = h
	fmt.Fprintf(stderrLog, "hybrid: loaded %d chunks (model=%s dim=%d)\n", h.Len(), h.EmbModel, h.Dim)
}
