package retrieval

import "testing"

func buildIndex() *BM25Index {
	idx := NewBM25()
	idx.Add("d1", "the unholy DK rotation uses Soul Reaper at 35% health")
	idx.Add("d2", "interrupt priority for spell 452030 is critical in the dungeon")
	idx.Add("d3", "C_EncounterTimeline ETEA event fires for boss casts not trash")
	idx.Add("d4", "the calendar API uses EventSignUp not EventAvailable")
	return idx
}

func TestBM25ExactNumericMatch(t *testing.T) {
	idx := buildIndex()
	// A spell ID is the canonical "embeddings would blur this" case.
	res := idx.Search("452030", 5)
	if len(res) == 0 || res[0].ID != "d2" {
		t.Fatalf("spell id 452030 should rank d2 first, got %+v", res)
	}
}

func TestBM25IdentifierMatch(t *testing.T) {
	idx := buildIndex()
	res := idx.Search("C_EncounterTimeline", 5)
	if len(res) == 0 || res[0].ID != "d3" {
		t.Fatalf("C_EncounterTimeline should rank d3 first, got %+v", res)
	}
}

func TestBM25RankingAndTopK(t *testing.T) {
	idx := buildIndex()
	res := idx.Search("rotation Soul Reaper", 1)
	if len(res) != 1 {
		t.Fatalf("top-k=1 should return exactly 1 result, got %d", len(res))
	}
	if res[0].ID != "d1" {
		t.Fatalf("rotation/Soul Reaper should rank d1 first, got %s", res[0].ID)
	}
}

func TestBM25NoMatchReturnsEmpty(t *testing.T) {
	idx := buildIndex()
	if res := idx.Search("transmogrification pet battles", 5); len(res) != 0 {
		t.Fatalf("query with no overlapping terms should return empty, got %+v", res)
	}
}

func TestBM25EmptyIndex(t *testing.T) {
	if res := NewBM25().Search("anything", 5); res != nil {
		t.Fatalf("empty index must return nil, got %+v", res)
	}
}

func TestTokenizePreservesIDsAndSplitsIdentifiers(t *testing.T) {
	got := tokenize("C_EncounterTimeline spell=452030!!")
	want := []string{"c", "encountertimeline", "spell", "452030"}
	if len(got) != len(want) {
		t.Fatalf("token count: want %v got %v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: want %q got %q", i, want[i], got[i])
		}
	}
}
