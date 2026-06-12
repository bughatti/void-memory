package retrieval

import "testing"

func TestRRFRewardsAgreement(t *testing.T) {
	// d2 ranks high in BOTH lists; d1 and d3 each rank high in only one.
	lexical := []Scored{{"d2", 9}, {"d1", 5}, {"d3", 1}}
	vector := []Scored{{"d2", 0.9}, {"d3", 0.8}, {"d1", 0.2}}
	out := RRF([][]Scored{lexical, vector}, 0, DefaultRRFK)
	if out[0].ID != "d2" {
		t.Fatalf("doc ranking high in both lists should win, got %+v", out)
	}
}

func TestRRFUnionOfCandidates(t *testing.T) {
	// A doc present in only one list still appears in the fused output.
	lexical := []Scored{{"only-lex", 3}}
	vector := []Scored{{"only-vec", 0.5}}
	out := RRF([][]Scored{lexical, vector}, 0, DefaultRRFK)
	if len(out) != 2 {
		t.Fatalf("fused output should union both lists, got %+v", out)
	}
}

func TestRRFTopK(t *testing.T) {
	lexical := []Scored{{"a", 3}, {"b", 2}, {"c", 1}}
	vector := []Scored{{"a", 3}, {"b", 2}, {"c", 1}}
	out := RRF([][]Scored{lexical, vector}, 2, DefaultRRFK)
	if len(out) != 2 {
		t.Fatalf("top-k=2 should cap output at 2, got %d", len(out))
	}
}

func TestRRFEmpty(t *testing.T) {
	if out := RRF(nil, 5, DefaultRRFK); len(out) != 0 {
		t.Fatalf("fusing no lists should be empty, got %+v", out)
	}
}
