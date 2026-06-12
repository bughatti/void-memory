// index.go is the local backend's HybridIndex: the in-process store that holds
// chunks, the BM25 lexical index, and per-chunk binary + float vectors, and
// runs the full hybrid search. It is pure (no Ollama / no disk in Search) so it
// unit-tests with injected embeddings; the ingestion layer (build.go) supplies
// real vectors from the Ollama embedder, and Save/Load persist it under
// ~/.void-memory/.
//
// Search pipeline (per REBUILD-PLAN.md):
//   query vec ─┬─► binary Hamming shortlist ─► float cosine RESCORE ─► semantic rank
//   query text ┴─► BM25 ───────────────────────────────────────────► lexical rank
//                                    │
//                                    └─► RRF fuse ─► top-K chunks ─► (LLM synthesis)
package retrieval

import (
	"encoding/gob"
	"os"
	"path/filepath"
	"sort"
)

// HybridIndex stores chunks and their lexical + vector representations.
// Not safe for concurrent Add+Search; the ingestion phase populates it single-
// threaded, then it is read at query time.
type HybridIndex struct {
	EmbModel string       // embedding model the vectors were produced with
	Dim      int          // embedding dimensionality (0 until first Add)
	Chunks   []Chunk      // parallel to Codes/Floats
	Codes    [][]uint64   // binary-quantized vectors
	Floats   [][]float32  // retained float vectors for rescore

	bm25 *BM25Index // rebuilt from Chunks on Load (not gob-encoded)
}

// NewHybridIndex returns an empty index for the given embedding model.
func NewHybridIndex(embModel string) *HybridIndex {
	return &HybridIndex{EmbModel: embModel, bm25: NewBM25()}
}

// Len returns the number of indexed chunks.
func (h *HybridIndex) Len() int { return len(h.Chunks) }

// Add inserts one chunk with its embedding. The first Add fixes Dim; later
// vectors must match it. Returns false if the vector dimensionality mismatches
// (the chunk is skipped rather than corrupting the index).
func (h *HybridIndex) Add(c Chunk, vec []float32) bool {
	if h.bm25 == nil {
		h.bm25 = NewBM25()
	}
	if h.Dim == 0 {
		h.Dim = len(vec)
	}
	if len(vec) != h.Dim || h.Dim == 0 {
		return false
	}
	h.Chunks = append(h.Chunks, c)
	h.Codes = append(h.Codes, Quantize(vec))
	fcopy := make([]float32, len(vec))
	copy(fcopy, vec)
	h.Floats = append(h.Floats, fcopy)
	h.bm25.Add(c.ID, c.Text)
	return true
}

// SearchOptions tunes the hybrid search. Zero value uses defaults.
type SearchOptions struct {
	Shortlist int     // candidates pulled from each retriever before fusion (default 100)
	RRFK      float64 // RRF constant (default DefaultRRFK)
}

func (o SearchOptions) withDefaults() SearchOptions {
	if o.Shortlist <= 0 {
		o.Shortlist = 100
	}
	if o.RRFK <= 0 {
		o.RRFK = DefaultRRFK
	}
	return o
}

// Result is one ranked chunk with its fused score.
type Result struct {
	Chunk Chunk
	Score float64
}

// Search runs hybrid retrieval for a query given its precomputed embedding and
// its raw text, returning the top-k chunks. queryVec may be nil (lexical-only,
// e.g. when the embedder is unavailable) — BM25 still works.
func (h *HybridIndex) Search(queryVec []float32, queryText string, k int, opts SearchOptions) []Result {
	opts = opts.withDefaults()
	if len(h.Chunks) == 0 {
		return nil
	}

	lists := make([][]Scored, 0, 2)

	// Lexical half.
	if h.bm25 != nil {
		if lex := h.bm25.Search(queryText, opts.Shortlist); len(lex) > 0 {
			lists = append(lists, lex)
		}
	}

	// Semantic half: binary Hamming shortlist, then float cosine rescore.
	if len(queryVec) == h.Dim && h.Dim > 0 {
		lists = append(lists, h.semanticRanked(queryVec, opts.Shortlist))
	}

	fused := RRF(lists, k, opts.RRFK)
	out := make([]Result, 0, len(fused))
	idx := h.idByID()
	for _, s := range fused {
		if i, ok := idx[s.ID]; ok {
			out = append(out, Result{Chunk: h.Chunks[i], Score: s.Score})
		}
	}
	return out
}

// semanticRanked returns chunks ranked by the binary→float rescore pipeline:
// shortlist by Hamming distance (cheap), then re-rank that shortlist by exact
// float cosine (accurate). This is what recovers ~96% of full-precision recall
// at ~1/32 the search cost.
func (h *HybridIndex) semanticRanked(queryVec []float32, shortlist int) []Scored {
	qcode := Quantize(queryVec)

	type cand struct {
		i    int
		dist int
	}
	cands := make([]cand, len(h.Codes))
	for i, code := range h.Codes {
		cands[i] = cand{i: i, dist: Hamming(qcode, code)}
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].dist < cands[b].dist })
	if len(cands) > shortlist {
		cands = cands[:shortlist]
	}

	rescored := make([]Scored, len(cands))
	for n, c := range cands {
		rescored[n] = Scored{ID: h.Chunks[c.i].ID, Score: Cosine(queryVec, h.Floats[c.i])}
	}
	sort.Slice(rescored, func(a, b int) bool {
		if rescored[a].Score != rescored[b].Score {
			return rescored[a].Score > rescored[b].Score
		}
		return rescored[a].ID < rescored[b].ID
	})
	return rescored
}

func (h *HybridIndex) idByID() map[string]int {
	m := make(map[string]int, len(h.Chunks))
	for i, c := range h.Chunks {
		m[c.ID] = i
	}
	return m
}

// Save persists the index to path (gob). The BM25 index is NOT serialized — it
// is deterministically rebuilt from Chunks on Load, keeping the on-disk format
// small and version-tolerant. Writes atomically via a temp file + rename.
func (h *HybridIndex) Save(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := gob.NewEncoder(f).Encode(h); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// LoadHybridIndex reads an index from path and rebuilds its BM25 index.
func LoadHybridIndex(path string) (*HybridIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var h HybridIndex
	if err := gob.NewDecoder(f).Decode(&h); err != nil {
		return nil, err
	}
	h.bm25 = NewBM25()
	for _, c := range h.Chunks {
		h.bm25.Add(c.ID, c.Text)
	}
	return &h, nil
}
