package retrieval

import (
	"math"
	"testing"
)

func TestQuantizeSignBits(t *testing.T) {
	// dims 0,2 positive -> bits 0 and 2 set => 0b101 = 5
	vec := []float32{0.7, -0.1, 0.001, -3.0}
	code := Quantize(vec)
	if len(code) != 1 {
		t.Fatalf("want 1 word for 4 dims, got %d", len(code))
	}
	if code[0] != 0b101 {
		t.Fatalf("want bits 0,2 set (0b101=5), got %b", code[0])
	}
}

func TestQuantizeWordBoundary(t *testing.T) {
	// 65 dims must span 2 words; bit 64 lands in word 1, bit 0.
	vec := make([]float32, 65)
	vec[64] = 1.0
	code := Quantize(vec)
	if len(code) != 2 {
		t.Fatalf("want 2 words for 65 dims, got %d", len(code))
	}
	if code[0] != 0 || code[1] != 1 {
		t.Fatalf("bit 64 should be word1/bit0; got w0=%b w1=%b", code[0], code[1])
	}
}

func TestHammingIdenticalAndOpposite(t *testing.T) {
	a := Quantize([]float32{1, 1, 1, 1})
	if d := Hamming(a, a); d != 0 {
		t.Fatalf("identical codes must have Hamming 0, got %d", d)
	}
	b := Quantize([]float32{-1, -1, -1, -1})
	if d := Hamming(a, b); d != 4 {
		t.Fatalf("opposite 4-dim codes must differ in 4 bits, got %d", d)
	}
}

func TestHammingSimilarity(t *testing.T) {
	if s := HammingSimilarity(0, 64); s != 1.0 {
		t.Fatalf("dist 0 must be similarity 1.0, got %v", s)
	}
	if s := HammingSimilarity(64, 64); s != 0.0 {
		t.Fatalf("all-bits-differ must be 0.0, got %v", s)
	}
	if s := HammingSimilarity(16, 64); math.Abs(s-0.75) > 1e-9 {
		t.Fatalf("16/64 differ must be 0.75, got %v", s)
	}
}

func TestCosine(t *testing.T) {
	a := []float32{1, 0, 0}
	if c := Cosine(a, a); math.Abs(c-1.0) > 1e-9 {
		t.Fatalf("self-cosine must be 1.0, got %v", c)
	}
	b := []float32{0, 1, 0}
	if c := Cosine(a, b); math.Abs(c) > 1e-9 {
		t.Fatalf("orthogonal cosine must be 0, got %v", c)
	}
	c := []float32{-1, 0, 0}
	if got := Cosine(a, c); math.Abs(got+1.0) > 1e-9 {
		t.Fatalf("opposite cosine must be -1.0, got %v", got)
	}
}

// TestBinaryPreservesNearestNeighbor is a sanity check that binary Hamming
// ranking agrees with float cosine for an easy case: the binary nearest
// neighbor of a query should be the same vector float cosine prefers.
func TestBinaryPreservesNearestNeighbor(t *testing.T) {
	query := []float32{0.9, 0.8, -0.2, 0.1, -0.7}
	near := []float32{0.7, 0.6, -0.1, 0.3, -0.9}  // same sign pattern
	far := []float32{-0.9, -0.8, 0.5, -0.4, 0.6}  // opposite-ish signs

	qc, nc, fc := Quantize(query), Quantize(near), Quantize(far)
	if Hamming(qc, nc) >= Hamming(qc, fc) {
		t.Fatalf("near vector should have smaller Hamming than far")
	}
	if Cosine(query, near) <= Cosine(query, far) {
		t.Fatalf("near vector should have larger cosine than far")
	}
}
