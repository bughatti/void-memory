package retrieval

import (
	"path/filepath"
	"testing"
)

func ck(id, text string) Chunk { return Chunk{ID: id, SessionID: "s", Text: text} }

// three chunks with deliberately distinct sign patterns so binary Hamming
// cleanly separates them.
func seedIndex() *HybridIndex {
	h := NewHybridIndex("test")
	h.Add(ck("a", "unholy dk soul reaper rotation"), []float32{1, 1, 1, 1, -1, -1, -1, -1})
	h.Add(ck("b", "interrupt spell 452030 in the dungeon"), []float32{-1, -1, -1, -1, 1, 1, 1, 1})
	h.Add(ck("c", "calendar api EventSignUp flow"), []float32{1, -1, 1, -1, 1, -1, 1, -1})
	return h
}

func TestHybridFusionSemanticAndLexicalAgree(t *testing.T) {
	h := seedIndex()
	// Query vector close to "a", text also about a's topic -> a wins decisively.
	qv := []float32{0.9, 0.8, 0.7, 0.6, -0.5, -0.6, -0.7, -0.8}
	res := h.Search(qv, "soul reaper rotation", 3, SearchOptions{})
	if len(res) == 0 || res[0].Chunk.ID != "a" {
		t.Fatalf("semantic+lexical agreement should rank 'a' first, got %+v", res)
	}
}

func TestHybridLexicalOnlyExactID(t *testing.T) {
	h := seedIndex()
	// nil query vector -> lexical-only path; exact spell-id match wins.
	res := h.Search(nil, "452030", 3, SearchOptions{})
	if len(res) == 0 || res[0].Chunk.ID != "b" {
		t.Fatalf("lexical-only exact id 452030 should rank 'b' first, got %+v", res)
	}
}

func TestHybridDimMismatchSkipped(t *testing.T) {
	h := seedIndex()
	before := h.Len()
	if ok := h.Add(ck("bad", "wrong dims"), []float32{1, 2, 3}); ok {
		t.Fatalf("dim-mismatched add should return false")
	}
	if h.Len() != before {
		t.Fatalf("dim-mismatched chunk must not be added; len changed %d->%d", before, h.Len())
	}
}

func TestHybridSaveLoadRoundTrip(t *testing.T) {
	h := seedIndex()
	path := filepath.Join(t.TempDir(), "index.gob")
	if err := h.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	loaded, err := LoadHybridIndex(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.Len() != h.Len() || loaded.Dim != h.Dim || loaded.EmbModel != h.EmbModel {
		t.Fatalf("metadata not preserved: len %d/%d dim %d/%d model %q/%q",
			loaded.Len(), h.Len(), loaded.Dim, h.Dim, loaded.EmbModel, h.EmbModel)
	}
	// BM25 must be rebuilt on load: lexical search still works.
	res := loaded.Search(nil, "452030", 1, SearchOptions{})
	if len(res) == 0 || res[0].Chunk.ID != "b" {
		t.Fatalf("post-load lexical search broken, got %+v", res)
	}
}

func TestHybridEmptyIndex(t *testing.T) {
	if res := NewHybridIndex("test").Search([]float32{1}, "q", 5, SearchOptions{}); res != nil {
		t.Fatalf("empty index must return nil, got %+v", res)
	}
}
