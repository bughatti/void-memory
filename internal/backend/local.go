// local.go implements MemoryBackend against the local filesystem + local
// Ollama. Retrieval is hybrid (BM25 + binary-quantized vectors over chunked
// transcripts); the LLM is used only for query-time synthesis. The legacy
// LLM router, metadata extractor, and catalog were removed — the hybrid index
// is the single source of truth, and parser.go provides transcript reading.
package backend

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/bughatti/void-memory/internal/indexer"
	"github.com/bughatti/void-memory/internal/ollama"
	"github.com/bughatti/void-memory/internal/recall"
	"github.com/bughatti/void-memory/internal/retrieval"
	"github.com/bughatti/void-memory/pkg/types"
)

// LocalBackend serves recall from a local hybrid index + local Ollama.
type LocalBackend struct {
	syn   *recall.Synthesizer
	llm   *ollama.Client
	model string

	dataDir     string
	projectsDir string
	embModel    string
	hybrid      *retrieval.HybridIndex
}

// LocalConfig holds the config the LocalBackend needs at construction.
type LocalConfig struct {
	DataDir     string // ~/.void-memory by default
	ProjectsDir string // ~/.claude/projects
	OllamaURL   string // "" → localhost:11434
	Model       string // synthesis model, e.g. "qwen2.5-coder:3b"
	EmbModel    string // embedding model, e.g. "nomic-embed-text"
}

// NewLocal constructs the backend and loads the persisted hybrid index (if any).
// A missing index is not an error — recall returns empty until `reindex` runs.
func NewLocal(cfg LocalConfig) (*LocalBackend, error) {
	if cfg.Model == "" {
		cfg.Model = "qwen2.5-coder:3b"
	}
	if cfg.EmbModel == "" {
		cfg.EmbModel = "nomic-embed-text"
	}
	llm := ollama.New(cfg.OllamaURL)
	// Don't hard-fail if Ollama is briefly unreachable at startup — recall and
	// reindex surface errors when called. Keeps the MCP server alive across
	// Ollama restarts.
	_, _ = llm.Version(context.Background())

	b := &LocalBackend{
		syn:         recall.NewSynthesizer(llm, cfg.Model),
		llm:         llm,
		model:       cfg.Model,
		dataDir:     cfg.DataDir,
		projectsDir: cfg.ProjectsDir,
		embModel:    cfg.EmbModel,
	}
	b.loadHybridIndex()
	return b, nil
}

// stderrLog routes diagnostic output to STDERR. Critical: when running as the
// MCP server, stdout carries the JSON-RPC protocol stream, so any diagnostic on
// stdout would corrupt it.
var stderrLog = (interface {
	Write([]byte) (int, error)
})(stderrWriter{})

type stderrWriter struct{}

func (stderrWriter) Write(b []byte) (int, error) {
	return fmt.Fprint(os.Stderr, string(b))
}

// Recall implements MemoryBackend. Hybrid retrieval is the gate; the LLM is
// deferred to synthesis over the small retrieved set.
func (b *LocalBackend) Recall(ctx context.Context, query string, hints RecallHints) (*types.RecallResult, error) {
	if b.hybrid != nil && b.hybrid.Len() > 0 {
		return b.hybridRecall(ctx, query, hints)
	}
	// No index loaded — return empty. Run `void-memory reindex` to build it.
	return &types.RecallResult{GeneratedAt: time.Now().UTC()}, nil
}

// ReadSession implements MemoryBackend. Maps a session id (the JSONL filename
// stem) back to its transcript by scanning the projects dir — no catalog needed.
func (b *LocalBackend) ReadSession(ctx context.Context, sessionID string) (*types.SessionTranscript, error) {
	paths, err := indexer.EnumerateProjectsDir(b.projectsDir)
	if err != nil {
		return nil, fmt.Errorf("enumerate transcripts: %w", err)
	}
	for _, p := range paths {
		if strings.TrimSuffix(filepath.Base(p), ".jsonl") == sessionID {
			ps, err := indexer.ParseFile(p)
			if err != nil {
				return nil, fmt.Errorf("parse %s: %w", p, err)
			}
			return &types.SessionTranscript{
				SessionID: ps.SessionID,
				Path:      ps.Path,
				Messages:  ps.Messages,
			}, nil
		}
	}
	return nil, fmt.Errorf("session %s not found under %s", sessionID, b.projectsDir)
}

// ListTopics implements MemoryBackend. Derives the session list from the hybrid
// index (distinct sessions + their chunk counts + time spans). The category
// filter is a legacy concept and is ignored.
func (b *LocalBackend) ListTopics(ctx context.Context, category string) ([]types.SessionMeta, error) {
	if b.hybrid == nil {
		return nil, nil
	}
	type agg struct {
		start, end time.Time
		n          int
	}
	m := map[string]*agg{}
	for _, c := range b.hybrid.Chunks {
		a := m[c.SessionID]
		if a == nil {
			a = &agg{start: c.StartTime, end: c.EndTime}
			m[c.SessionID] = a
		}
		if !c.StartTime.IsZero() && (a.start.IsZero() || c.StartTime.Before(a.start)) {
			a.start = c.StartTime
		}
		if c.EndTime.After(a.end) {
			a.end = c.EndTime
		}
		a.n++
	}
	out := make([]types.SessionMeta, 0, len(m))
	for id, a := range m {
		out = append(out, types.SessionMeta{SessionID: id, StartTime: a.start, EndTime: a.end, ChunkCount: a.n})
	}
	return out, nil
}

// IndexStatus implements MemoryBackend, reporting from the hybrid index.
func (b *LocalBackend) IndexStatus(ctx context.Context) (*Status, error) {
	sessions := map[string]struct{}{}
	if b.hybrid != nil {
		for _, c := range b.hybrid.Chunks {
			sessions[c.SessionID] = struct{}{}
		}
	}
	reachable := true
	if _, err := b.llm.Version(ctx); err != nil {
		reachable = false
	}
	last := "never"
	if fi, err := os.Stat(hybridIndexPath(b.dataDir)); err == nil {
		last = fi.ModTime().UTC().Format(time.RFC3339)
	}
	return &Status{
		Backend:         "local",
		SessionCount:    len(sessions),
		LastIndexedAt:   last,
		ModelInUse:      b.model,
		OllamaReachable: reachable,
		IndexerQueueLen: 0,
	}, nil
}

// Close implements MemoryBackend.
func (b *LocalBackend) Close() error { return nil }
