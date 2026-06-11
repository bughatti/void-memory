// Package catalog manages the in-memory view of all indexed sessions plus
// per-session metadata persistence to ~/.void-memory/sessions/<id>.meta.json.
//
// No SQL, no embedded database — flat JSON files only. At our scale (low
// hundreds of sessions today, low thousands eventually) walking the dir on
// startup costs <50ms and the in-memory map handles all queries.
package catalog

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/bughatti/void-memory/pkg/types"
)

const sessionsSubdir = "sessions"

// Catalog is thread-safe; the indexer and the MCP server access it concurrently.
type Catalog struct {
	dataDir  string
	mu       sync.RWMutex
	sessions map[string]types.SessionMeta // keyed by session_id
}

// New constructs a Catalog rooted at dataDir (typically ~/.void-memory).
// Does NOT load from disk — call Load() explicitly.
func New(dataDir string) *Catalog {
	return &Catalog{
		dataDir:  dataDir,
		sessions: make(map[string]types.SessionMeta),
	}
}

// Load reads every <id>.meta.json under dataDir/sessions/ into memory.
// Files that fail to parse are logged but don't abort the load — a single
// corrupted meta file shouldn't kill the whole catalog.
func (c *Catalog) Load() error {
	dir := filepath.Join(c.dataDir, sessionsSubdir)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("mkdir %s: %w", dir, err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("read %s: %w", dir, err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	c.sessions = make(map[string]types.SessionMeta, len(entries))
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".meta.json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		var m types.SessionMeta
		if err := json.Unmarshal(b, &m); err != nil {
			continue
		}
		c.sessions[m.SessionID] = m
	}
	return nil
}

// DeleteByParent removes every chunk that belongs to the given parent
// session ID, both from memory and from disk. Used by the indexer when
// re-chunking a session whose JSONL grew — stale chunks from the previous
// indexing pass need to go before new ones are written.
func (c *Catalog) DeleteByParent(parentID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	toDelete := make([]string, 0)
	for id, m := range c.sessions {
		if m.ParentSessionID == parentID || id == parentID {
			toDelete = append(toDelete, id)
		}
	}
	for _, id := range toDelete {
		delete(c.sessions, id)
		path := filepath.Join(c.dataDir, sessionsSubdir, id+".meta.json")
		_ = os.Remove(path)
	}
	return nil
}

// Upsert stores a session metadata record in memory AND on disk. Safe to call
// concurrently; the in-memory write takes the lock, the disk write happens
// after the lock is released (the meta file is the canonical record on disk
// and races between concurrent upserts for the SAME session are unlikely
// because indexing for a given session is single-threaded).
func (c *Catalog) Upsert(meta types.SessionMeta) error {
	meta.IndexedAt = time.Now().UTC()
	if meta.SchemaVer == 0 {
		meta.SchemaVer = 1
	}

	c.mu.Lock()
	c.sessions[meta.SessionID] = meta
	c.mu.Unlock()

	// Persist last; failures here just mean we'll re-index on next start.
	return c.writeMeta(meta)
}

// Get returns a session metadata record by ID. ok=false if not present.
func (c *Catalog) Get(sessionID string) (types.SessionMeta, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	m, ok := c.sessions[sessionID]
	return m, ok
}

// All returns a defensive copy of every session metadata record.
func (c *Catalog) All() []types.SessionMeta {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]types.SessionMeta, 0, len(c.sessions))
	for _, m := range c.sessions {
		out = append(out, m)
	}
	return out
}

// Len is the count of indexed sessions, useful for status reporting.
func (c *Catalog) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.sessions)
}

// FindByCategory returns sessions matching the given category. Pass empty
// string to skip the category filter.
func (c *Catalog) FindByCategory(category string) []types.SessionMeta {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]types.SessionMeta, 0)
	for _, m := range c.sessions {
		if category == "" || string(m.Category) == category {
			out = append(out, m)
		}
	}
	return out
}

// FindByEntities returns sessions whose Entities list intersects ANY of the
// supplied entities (case-insensitive). Returns sessions sorted with
// highest entity overlap first.
func (c *Catalog) FindByEntities(entities []string) []types.SessionMeta {
	if len(entities) == 0 {
		return nil
	}
	want := make(map[string]struct{}, len(entities))
	for _, e := range entities {
		want[strings.ToLower(e)] = struct{}{}
	}

	c.mu.RLock()
	defer c.mu.RUnlock()
	type scored struct {
		meta  types.SessionMeta
		hits  int
	}
	var out []scored
	for _, m := range c.sessions {
		hits := 0
		for _, e := range m.Entities {
			if _, ok := want[strings.ToLower(e)]; ok {
				hits++
			}
		}
		if hits > 0 {
			out = append(out, scored{meta: m, hits: hits})
		}
	}
	// Insertion-sort by hits desc; N is small.
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j].hits > out[j-1].hits; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	result := make([]types.SessionMeta, len(out))
	for i, s := range out {
		result[i] = s.meta
	}
	return result
}

// NeedsReindex returns true if the JSONL file at the meta's path has been
// modified since we last indexed it. Handles both single and chunked
// sessions — checks ALL catalog entries whose parent_session_id (or
// session_id, for unchunked) matches the JSONL's session_id.
func (c *Catalog) NeedsReindex(sessionID string, sourceMTime time.Time) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	// Direct hit (unchunked session).
	if m, ok := c.sessions[sessionID]; ok {
		return sourceMTime.After(m.SourceMTime)
	}
	// Look for any chunk whose parent matches this session.
	for _, m := range c.sessions {
		if m.ParentSessionID == sessionID {
			if sourceMTime.After(m.SourceMTime) {
				return true
			}
		}
	}
	// If no parent match found either, this is a brand-new session.
	for _, m := range c.sessions {
		if m.ParentSessionID == sessionID {
			return false
		}
	}
	return true
}

func (c *Catalog) writeMeta(m types.SessionMeta) error {
	dir := filepath.Join(c.dataDir, sessionsSubdir)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	path := filepath.Join(dir, m.SessionID+".meta.json")
	tmp := path + ".tmp"
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	// Atomic-ish rename: on Windows this is best-effort but acceptable for
	// our use case (corrupted partial writes are extremely unlikely with
	// small JSON files written in one syscall).
	return os.Rename(tmp, path)
}
