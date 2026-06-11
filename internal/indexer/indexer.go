// indexer.go is the runtime loop: discover sessions, parse them, extract
// metadata via the LLM, upsert into the catalog. Also watches for new
// JSONLs via fsnotify so newly-saved sessions get indexed in near-realtime.
package indexer

import (
	"context"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/bughatti/void-memory/internal/catalog"
	"github.com/bughatti/void-memory/pkg/types"
	"github.com/fsnotify/fsnotify"
)

// Indexer wires Catalog + Extractor + the projects directory watcher.
type Indexer struct {
	cat        *catalog.Catalog
	ext        *Extractor
	projectsDir string

	mu     sync.Mutex
	queue  map[string]struct{} // session_id → enqueued
	queueCh chan string

	queueLen int32
}

// New constructs an Indexer. projectsDir should be the path to
// ~/.claude/projects (NOT a specific project subdir — we walk all).
func New(cat *catalog.Catalog, ext *Extractor, projectsDir string) *Indexer {
	return &Indexer{
		cat:         cat,
		ext:         ext,
		projectsDir: projectsDir,
		queue:       make(map[string]struct{}),
		queueCh:     make(chan string, 64),
	}
}

// Run starts the indexer loop. Blocks until ctx is canceled. The loop:
//   1. Discovers all existing JSONLs in projectsDir, enqueues any that
//      need (re)indexing.
//   2. Watches projectsDir for new files / changes via fsnotify, enqueues
//      detected changes.
//   3. A worker goroutine pulls from the queue and runs Extract, upserting
//      to the catalog.
func (idx *Indexer) Run(ctx context.Context) error {
	w, err := fsnotify.NewWatcher()
	if err != nil {
		return err
	}
	defer w.Close()
	if err := w.Add(idx.projectsDir); err != nil {
		return err
	}
	// Watch every project subdir too — top-level entries may be added later.
	entries, _ := os.ReadDir(idx.projectsDir)
	for _, e := range entries {
		if e.IsDir() {
			_ = w.Add(filepath.Join(idx.projectsDir, e.Name()))
		}
	}

	// Initial discovery pass.
	go idx.discover()

	// Worker.
	go idx.worker(ctx)

	for {
		select {
		case <-ctx.Done():
			return nil
		case ev, ok := <-w.Events:
			if !ok {
				return nil
			}
			idx.handleEvent(w, ev)
		case err, ok := <-w.Errors:
			if !ok {
				return nil
			}
			log.Printf("fsnotify error: %v", err)
		}
	}
}

func (idx *Indexer) handleEvent(w *fsnotify.Watcher, ev fsnotify.Event) {
	// Add new project dirs to the watch list.
	if ev.Has(fsnotify.Create) {
		st, err := os.Stat(ev.Name)
		if err == nil && st.IsDir() {
			_ = w.Add(ev.Name)
			return
		}
	}
	// Only react to JSONLs.
	if !strings.HasSuffix(ev.Name, ".jsonl") {
		return
	}
	// Skip subagent / workflow JSONLs — they're not user conversations.
	if isSubagentPath(ev.Name) {
		return
	}
	if ev.Has(fsnotify.Create) || ev.Has(fsnotify.Write) {
		sessionID := strings.TrimSuffix(filepath.Base(ev.Name), ".jsonl")
		idx.enqueue(sessionID, ev.Name)
	}
}

func isSubagentPath(p string) bool {
	return strings.Contains(filepath.ToSlash(p), "/subagents/")
}

func (idx *Indexer) discover() {
	paths, err := EnumerateProjectsDir(idx.projectsDir)
	if err != nil {
		log.Printf("discover: %v", err)
		return
	}
	for _, p := range paths {
		st, err := os.Stat(p)
		if err != nil {
			continue
		}
		sessionID := strings.TrimSuffix(filepath.Base(p), ".jsonl")
		if idx.cat.NeedsReindex(sessionID, st.ModTime()) {
			idx.enqueue(sessionID, p)
		}
	}
}

func (idx *Indexer) enqueue(sessionID, path string) {
	idx.mu.Lock()
	if _, dup := idx.queue[sessionID]; dup {
		idx.mu.Unlock()
		return
	}
	idx.queue[sessionID] = struct{}{}
	idx.queueLen++
	idx.mu.Unlock()

	// Non-blocking send: if the channel is full, the worker will pick it up
	// on the next discovery pass via NeedsReindex.
	select {
	case idx.queueCh <- path:
	default:
	}
}

func (idx *Indexer) dequeue(sessionID string) {
	idx.mu.Lock()
	delete(idx.queue, sessionID)
	if idx.queueLen > 0 {
		idx.queueLen--
	}
	idx.mu.Unlock()
}

// QueueLen reports queue depth for status endpoints.
func (idx *Indexer) QueueLen() int {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return int(idx.queueLen)
}

func (idx *Indexer) worker(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case path, ok := <-idx.queueCh:
			if !ok {
				return
			}
			idx.processOne(ctx, path)
		}
	}
}

func (idx *Indexer) processOne(ctx context.Context, path string) {
	sessionID := strings.TrimSuffix(filepath.Base(path), ".jsonl")
	defer idx.dequeue(sessionID)

	ps, err := ParseFile(path)
	if err != nil {
		log.Printf("parse %s: %v", path, err)
		return
	}
	prompts := ps.UserPrompts()
	if len(prompts) == 0 {
		return
	}

	// Clear any prior chunks for this parent — a session that grew may have
	// different chunk boundaries now. Stale chunk records would otherwise
	// linger in the catalog.
	if err := idx.cat.DeleteByParent(ps.SessionID); err != nil {
		log.Printf("clear prior chunks for %s: %v", ps.SessionID, err)
	}

	// Long sessions get chunked into time-windowed sub-sessions so each
	// chunk gets its own category + entity extraction. Short sessions
	// skip this and go through as a single record (chunk index 0).
	var chunks []SessionChunk
	if ShouldChunk(ps) {
		chunks = ChunkSession(ps)
		log.Printf("chunking session=%s into %d chunks (%d prompts, span %v)",
			ps.SessionID, len(chunks), len(prompts), ps.EndTime.Sub(ps.StartTime))
	} else {
		chunks = ChunkSession(ps) // returns single chunk for short sessions
	}

	for _, c := range chunks {
		chunkPS := c.AsParsedSession()
		chunkPrompts := chunkPS.UserPrompts()
		if len(chunkPrompts) == 0 {
			continue
		}
		exCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
		meta, err := idx.ext.Extract(exCtx, chunkPS)
		cancel()
		if err != nil {
			log.Printf("extract %s: %v", chunkPS.SessionID, err)
			continue
		}
		meta.SessionID = chunkPS.SessionID
		meta.ParentSessionID = c.ParentSessionID
		meta.ChunkIndex = c.ChunkIndex
		meta.JSONLPath = chunkPS.Path
		meta.StartTime = chunkPS.StartTime
		meta.EndTime = chunkPS.EndTime
		meta.PromptCount = len(chunkPrompts)
		meta.SourceMTime = chunkPS.SourceMTime

		if err := idx.cat.Upsert(meta); err != nil {
			log.Printf("upsert %s: %v", chunkPS.SessionID, err)
			continue
		}
		log.Printf("indexed session=%s parent=%s chunk=%d category=%s entities=%d prompts=%d window=%s..%s",
			chunkPS.SessionID, c.ParentSessionID, c.ChunkIndex, meta.Category,
			len(meta.Entities), len(chunkPrompts),
			c.WindowStart.Format("2006-01-02"), c.WindowEnd.Format("2006-01-02"))
	}
}

// IndexNow runs a one-shot indexing pass synchronously and returns when all
// queued sessions are processed. Useful for the install/bootstrap step
// before the MCP server takes over.
func (idx *Indexer) IndexNow(ctx context.Context) error {
	paths, err := EnumerateProjectsDir(idx.projectsDir)
	if err != nil {
		return err
	}
	for _, p := range paths {
		st, err := os.Stat(p)
		if err != nil {
			continue
		}
		sessionID := strings.TrimSuffix(filepath.Base(p), ".jsonl")
		if !idx.cat.NeedsReindex(sessionID, st.ModTime()) {
			continue
		}
		idx.processOne(ctx, p)
	}
	return nil
}

// Stats returns a snapshot for the IndexStatus MCP tool.
type Stats struct {
	Sessions int
	QueueLen int
	LastRun  time.Time
}

func (idx *Indexer) Stats(cat *catalog.Catalog) Stats {
	// "Last run" inferred from the most recent IndexedAt across all metas.
	var last time.Time
	for _, m := range cat.All() {
		if m.IndexedAt.After(last) {
			last = m.IndexedAt
		}
	}
	return Stats{
		Sessions: cat.Len(),
		QueueLen: idx.QueueLen(),
		LastRun:  last,
	}
}

// silence unused-import linters for types.Message in non-debug builds
var _ = types.Message{}
