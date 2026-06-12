// bm25.go is the lexical half of hybrid retrieval: an in-memory inverted index
// with Okapi BM25 scoring. This is the zero-hallucination layer — it matches
// the user's actual tokens (spell IDs like 452030, API names like
// C_EncounterTimeline, "HARD RULE" keywords) exactly, where embeddings would
// blur them. Pure Go, no dependencies, fits the project's "no SQL" ethos at
// single-user corpus sizes (thousands of chunks).
//
// The same algorithm is mirrored by Postgres tsvector/GIN in the server
// backend; this in-memory version is the local backend's lexical index.
package retrieval

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// Default Okapi BM25 parameters (Lucene/Elasticsearch defaults).
const (
	defaultK1 = 1.2
	defaultB  = 0.75
)

type posting struct {
	doc int // internal doc index
	tf  int // term frequency in that doc
}

type bm25doc struct {
	id     string
	length int // token count
}

// BM25Index is an in-memory inverted index with BM25 scoring. Not safe for
// concurrent Add+Search; callers serialize writes (the indexer is single-
// goroutine) and may read after the write phase settles.
type BM25Index struct {
	k1, b    float64
	docs     []bm25doc
	postings map[string][]posting
	df       map[string]int // distinct-doc frequency per term
	totalLen int
}

// NewBM25 returns an empty index with default parameters.
func NewBM25() *BM25Index {
	return &BM25Index{
		k1:       defaultK1,
		b:        defaultB,
		postings: make(map[string][]posting),
		df:       make(map[string]int),
	}
}

// Len returns the number of indexed documents.
func (idx *BM25Index) Len() int { return len(idx.docs) }

// Add indexes one document under the given external id. Tokenization is
// lowercase runs of letters/digits, so identifiers and numeric IDs survive as
// whole tokens and match consistently between documents and queries.
func (idx *BM25Index) Add(id, text string) {
	toks := tokenize(text)
	docIdx := len(idx.docs)
	idx.docs = append(idx.docs, bm25doc{id: id, length: len(toks)})
	idx.totalLen += len(toks)

	tf := make(map[string]int, len(toks))
	for _, t := range toks {
		tf[t]++
	}
	for term, freq := range tf {
		idx.postings[term] = append(idx.postings[term], posting{doc: docIdx, tf: freq})
		idx.df[term]++
	}
}

// Scored is one ranked result: external id + BM25 score.
type Scored struct {
	ID    string
	Score float64
}

// Search returns the top-k documents for the query, ranked by BM25 score
// descending. k <= 0 returns all matched documents.
func (idx *BM25Index) Search(query string, k int) []Scored {
	n := len(idx.docs)
	if n == 0 {
		return nil
	}
	avgdl := float64(idx.totalLen) / float64(n)
	if avgdl == 0 {
		avgdl = 1
	}

	acc := make(map[int]float64)
	seen := make(map[string]bool)
	for _, term := range tokenize(query) {
		if seen[term] {
			continue // count each query term once
		}
		seen[term] = true
		plist, ok := idx.postings[term]
		if !ok {
			continue
		}
		idf := bm25IDF(n, idx.df[term])
		for _, p := range plist {
			dl := float64(idx.docs[p.doc].length)
			tf := float64(p.tf)
			denom := tf + idx.k1*(1-idx.b+idx.b*dl/avgdl)
			acc[p.doc] += idf * (tf * (idx.k1 + 1)) / denom
		}
	}

	out := make([]Scored, 0, len(acc))
	for doc, score := range acc {
		out = append(out, Scored{ID: idx.docs[doc].id, Score: score})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].ID < out[j].ID // stable tiebreak
	})
	if k > 0 && len(out) > k {
		out = out[:k]
	}
	return out
}

// bm25IDF is the Lucene-style BM25 IDF, which is always positive (the +1 inside
// the log prevents the negative-IDF pathology of the textbook formula for terms
// appearing in more than half the documents).
func bm25IDF(n, df int) float64 {
	return math.Log(1 + (float64(n)-float64(df)+0.5)/(float64(df)+0.5))
}

// tokenize lowercases and splits on any non-alphanumeric rune, yielding runs of
// [a-z0-9]. Numeric IDs (452030) and identifier fragments survive as tokens;
// "C_EncounterTimeline" -> ["c","encountertimeline"], matched the same way on
// the query side.
func tokenize(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}
