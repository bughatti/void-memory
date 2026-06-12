// fuse.go combines the lexical (BM25) and semantic (binary-vector) rankings
// into one ordered list using Reciprocal Rank Fusion (RRF). RRF is rank-based,
// not score-based, so it doesn't require the two retrievers' scores to be on
// comparable scales — it just rewards documents that rank highly in either
// list. This is the standard, robust way to fuse hybrid retrieval.
package retrieval

import "sort"

// DefaultRRFK is the standard RRF constant. Larger values flatten the
// contribution of top ranks; 60 is the widely-used default from the original
// Cormack et al. paper.
const DefaultRRFK = 60.0

// RRF fuses multiple ranked lists. Each input list must already be ordered
// best-first. A document's fused score is the sum over lists of
// 1/(rrfK + rank), with rank starting at 1. Returns the top-k fused results
// (k <= 0 returns all).
func RRF(lists [][]Scored, k int, rrfK float64) []Scored {
	if rrfK <= 0 {
		rrfK = DefaultRRFK
	}
	fused := make(map[string]float64)
	for _, list := range lists {
		for rank, s := range list {
			fused[s.ID] += 1.0 / (rrfK + float64(rank+1))
		}
	}
	out := make([]Scored, 0, len(fused))
	for id, score := range fused {
		out = append(out, Scored{ID: id, Score: score})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].ID < out[j].ID
	})
	if k > 0 && len(out) > k {
		out = out[:k]
	}
	return out
}
