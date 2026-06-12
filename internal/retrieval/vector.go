// Package retrieval implements the shared hybrid-retrieval core used by both
// the local and server backends: chunking, a BM25 lexical index, binary-
// quantized vector search, and RRF fusion. The design follows the rebuild
// research (see REBUILD-PLAN.md): the win is in the index/structure, so the
// heavy lifting is cheap CPU operations (Hamming popcount, inverted-index
// scoring) and the LLM is deferred to query-time synthesis over a small set.
//
// vector.go is the binary-quantization half. Embeddings (float32, from a small
// CPU model) are quantized to 1 bit/dimension and compared with Hamming
// distance (XOR + POPCNT, ~2 CPU cycles/word). This is 32x smaller and ~25x
// faster than float cosine on CPU, retaining ~96% accuracy when the top
// candidates are re-scored with the retained float vectors (see Rescore).
package retrieval

import (
	"math"
	"math/bits"
)

// Quantize converts a float32 embedding into a binary code, packed into
// uint64 words. Bit i is set when dimension i is > 0 (the standard sign-based
// binary quantization). The returned slice has ceil(len(vec)/64) words.
//
// Storing the sign bit is the whole trick: a 1024-dim float32 vector (4096
// bytes) becomes 16 uint64 words (128 bytes) — exactly 32x smaller.
func Quantize(vec []float32) []uint64 {
	words := (len(vec) + 63) / 64
	out := make([]uint64, words)
	for i, v := range vec {
		if v > 0 {
			out[i>>6] |= 1 << uint(i&63)
		}
	}
	return out
}

// Hamming returns the number of differing bits between two binary codes. This
// is the binary-space distance: smaller = more similar. Computed with XOR +
// OnesCount64, which the Go compiler lowers to a POPCNT instruction on amd64.
//
// Panics if the codes are different lengths — callers must compare codes from
// the same embedding model (same dimensionality).
func Hamming(a, b []uint64) int {
	if len(a) != len(b) {
		panic("retrieval: Hamming on codes of differing length")
	}
	d := 0
	for i := range a {
		d += bits.OnesCount64(a[i] ^ b[i])
	}
	return d
}

// HammingSimilarity maps Hamming distance to a [0,1] similarity score so it can
// be compared/fused with cosine. dim is the embedding dimensionality (number of
// bits). 1.0 = identical codes, 0.0 = every bit differs.
func HammingSimilarity(dist, dim int) float64 {
	if dim == 0 {
		return 0
	}
	return 1.0 - float64(dist)/float64(dim)
}

// Cosine is the float32 cosine similarity, used for the rescore stage over the
// small set of top binary candidates — this is what restores accuracy from
// ~92.5% (binary only) to ~96% (binary shortlist + float rescore).
func Cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
