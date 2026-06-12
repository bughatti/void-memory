// local.go implements MemoryBackend against the local filesystem +
// local Ollama. This is the only backend that ships in v1; ServerBackend
// (server.go, Phase 6) is the natural extension when multi-machine
// sharing is needed.
package backend

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/bughatti/void-memory/internal/catalog"
	"github.com/bughatti/void-memory/internal/indexer"
	"github.com/bughatti/void-memory/internal/ollama"
	"github.com/bughatti/void-memory/internal/recall"
	"github.com/bughatti/void-memory/internal/retrieval"
	"github.com/bughatti/void-memory/pkg/types"
)

// LocalBackend wires catalog + indexer + recall against local resources.
type LocalBackend struct {
	cat   *catalog.Catalog
	idx   *indexer.Indexer
	syn   *recall.Synthesizer
	llm   *ollama.Client
	model string

	// Hybrid-retrieval path (gated by config). When useHybrid is true and a
	// hybrid index is loaded, Recall uses it instead of the legacy LLM-routing
	// path. dataDir/projectsDir/embModel are retained for (re)indexing.
	dataDir     string
	projectsDir string
	embModel    string
	useHybrid   bool
	hybrid      *retrieval.HybridIndex

	cancel context.CancelFunc
}

// LocalConfig holds the config the LocalBackend needs at construction.
type LocalConfig struct {
	DataDir              string // ~/.void-memory by default
	ProjectsDir          string // ~/.claude/projects
	OllamaURL            string // "" → localhost:11434
	Model                string // e.g. "qwen2.5-coder:7b"
	Retrieval            string // "hybrid" enables the hybrid path; "" = legacy LLM-routing
	EmbModel             string // embedding model for hybrid (e.g. "nomic-embed-text")
	DisableBackgroundIdx bool   // CLI subcommands set this true so IndexNow doesn't race the watcher goroutine
}

// NewLocal constructs and starts the LocalBackend. The catalog is loaded
// from disk, the indexer is started in a background goroutine, and the
// router/synthesizer are wired against the configured Ollama. Returns an
// error if Ollama is unreachable or the catalog can't be loaded.
func NewLocal(cfg LocalConfig) (*LocalBackend, error) {
	if cfg.Model == "" {
		cfg.Model = "qwen2.5-coder:7b"
	}
	cat := catalog.New(cfg.DataDir)
	if err := cat.Load(); err != nil {
		return nil, fmt.Errorf("load catalog: %w", err)
	}
	llm := ollama.New(cfg.OllamaURL)

	// Don't hard-fail if Ollama is briefly unreachable at startup — the
	// indexer/recall steps will surface errors when they're called. This
	// keeps the MCP server alive across Ollama restarts.
	_, _ = llm.Version(context.Background())

	ext := indexer.NewExtractor(llm, cfg.Model)
	idx := indexer.New(cat, ext, cfg.ProjectsDir)

	ctx, cancel := context.WithCancel(context.Background())
	// The legacy LLM-based metadata indexer exists ONLY to serve the old
	// routing/scoping retrieval path. Hybrid retrieval doesn't use it, and it
	// would load the synthesis LLM onto the GPU at index time — so don't start
	// it in hybrid mode. (Full removal of the legacy path is scheduled once
	// hybrid is validated; see REBUILD-PLAN.md.)
	if !cfg.DisableBackgroundIdx && cfg.Retrieval != "hybrid" {
		go func() {
			if err := idx.Run(ctx); err != nil {
				fmt.Fprintf(stderrLog, "indexer stopped: %v\n", err)
			}
		}()
	}

	b := &LocalBackend{
		cat:         cat,
		idx:         idx,
		syn:         recall.NewSynthesizer(llm, cfg.Model),
		llm:         llm,
		model:       cfg.Model,
		dataDir:     cfg.DataDir,
		projectsDir: cfg.ProjectsDir,
		embModel:    cfg.EmbModel,
		useHybrid:   cfg.Retrieval == "hybrid",
		cancel:      cancel,
	}
	if b.useHybrid {
		b.loadHybridIndex()
	}
	return b, nil
}

// stderrLog routes diagnostic output to STDERR. This is critical: when running
// as the MCP server, stdout carries the JSON-RPC protocol stream, so any
// diagnostic written to stdout would corrupt it. (Previously this used
// fmt.Print → stdout, which is why hybrid's startup log broke MCP/JSON output.)
var stderrLog = (interface {
	Write([]byte) (int, error)
})(stderrWriter{})

type stderrWriter struct{}

func (stderrWriter) Write(b []byte) (int, error) {
	return fmt.Fprint(os.Stderr, string(b))
}

// Recall implements MemoryBackend. Retrieval (hybrid BM25 + binary-vector) is
// the gate; the LLM is deferred to synthesis over the small retrieved set. The
// legacy LLM-routing/scoping path has been removed.
func (b *LocalBackend) Recall(ctx context.Context, query string, hints RecallHints) (*types.RecallResult, error) {
	if b.hybrid != nil && b.hybrid.Len() > 0 {
		return b.hybridRecall(ctx, query, hints)
	}
	// Hybrid index not loaded — return empty rather than the removed legacy
	// path. Run `void-memory reindex` to build the index.
	return &types.RecallResult{GeneratedAt: time.Now().UTC()}, nil
}

// ReadSession implements MemoryBackend.
func (b *LocalBackend) ReadSession(ctx context.Context, sessionID string) (*types.SessionTranscript, error) {
	meta, ok := b.cat.Get(sessionID)
	if !ok {
		return nil, fmt.Errorf("session %s not in catalog", sessionID)
	}
	ps, err := indexer.ParseFile(meta.JSONLPath)
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", meta.JSONLPath, err)
	}
	return &types.SessionTranscript{
		SessionID: ps.SessionID,
		Path:      ps.Path,
		Messages:  ps.Messages,
	}, nil
}

// ListTopics implements MemoryBackend. category="" returns all.
func (b *LocalBackend) ListTopics(ctx context.Context, category string) ([]types.SessionMeta, error) {
	cat := strings.TrimSpace(category)
	if cat == "" {
		return b.cat.All(), nil
	}
	return b.cat.FindByCategory(cat), nil
}

// IndexStatus implements MemoryBackend.
func (b *LocalBackend) IndexStatus(ctx context.Context) (*Status, error) {
	stats := b.idx.Stats(b.cat)
	reachable := true
	if _, err := b.llm.Version(ctx); err != nil {
		reachable = false
	}
	last := stats.LastRun.Format(time.RFC3339)
	if stats.LastRun.IsZero() {
		last = "never"
	}
	return &Status{
		Backend:         "local",
		SessionCount:    stats.Sessions,
		LastIndexedAt:   last,
		ModelInUse:      b.model,
		OllamaReachable: reachable,
		IndexerQueueLen: stats.QueueLen,
	}, nil
}

// Close implements MemoryBackend.
func (b *LocalBackend) Close() error {
	if b.cancel != nil {
		b.cancel()
	}
	return nil
}

// IndexNow runs a synchronous indexing pass — used by the bootstrap step
// before the MCP server is fully serving.
func (b *LocalBackend) IndexNow(ctx context.Context) error {
	return b.idx.IndexNow(ctx)
}
