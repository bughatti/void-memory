package retrieval

import (
	"strings"
	"testing"
	"time"

	"github.com/bughatti/void-memory/pkg/types"
)

func msg(typ, content string, tool bool, result bool) types.Message {
	return types.Message{Type: typ, Content: content, IsToolUse: tool, IsToolResult: result, Timestamp: time.Now()}
}

func TestChunkPairsUserAndAssistant(t *testing.T) {
	msgs := []types.Message{
		msg("user", "how do I detect boss casts in 12.0.5?", false, false),
		msg("assistant", "Use C_EncounterTimeline ETEA events.", false, false),
		msg("user", "what about trash?", false, false),
		msg("assistant", "No clean path for trash spell IDs.", false, false),
	}
	chunks := ChunkMessages("sess1", msgs, ChunkOptions{})
	if len(chunks) != 2 {
		t.Fatalf("want 2 chunks (one per prompt), got %d", len(chunks))
	}
	if !strings.Contains(chunks[0].Text, "USER: how do I detect") || !strings.Contains(chunks[0].Text, "CLAUDE: Use C_EncounterTimeline") {
		t.Fatalf("chunk 0 should pair prompt+reply, got %q", chunks[0].Text)
	}
	if chunks[0].ID != "sess1#0" || chunks[1].ID != "sess1#1" {
		t.Fatalf("chunk ids wrong: %q %q", chunks[0].ID, chunks[1].ID)
	}
}

func TestChunkSkipsNoisePrompts(t *testing.T) {
	msgs := []types.Message{
		msg("user", "<system-reminder>blah</system-reminder>", false, false),
		msg("user", "<task-notification>done</task-notification>", false, false),
		msg("user", "real question here", false, false),
		msg("assistant", "real answer", false, false),
	}
	chunks := ChunkMessages("s", msgs, ChunkOptions{})
	if len(chunks) != 1 {
		t.Fatalf("noise prompts must not anchor chunks, got %d", len(chunks))
	}
	if !strings.Contains(chunks[0].Text, "real question") {
		t.Fatalf("expected the real prompt, got %q", chunks[0].Text)
	}
}

func TestChunkGathersMultipleAssistantTurnsAcrossTools(t *testing.T) {
	// Assistant text split across tool_use/tool_result should be gathered into
	// the same unit, with the tool turns skipped.
	msgs := []types.Message{
		msg("user", "build the thing", false, false),
		msg("assistant", "first I'll look", true, false),   // tool_use
		msg("user", "tool output", false, true),            // tool_result (noise)
		msg("assistant", "done, here's the result", false, false),
	}
	chunks := ChunkMessages("s", msgs, ChunkOptions{})
	if len(chunks) != 1 {
		t.Fatalf("want 1 chunk, got %d", len(chunks))
	}
	if !strings.Contains(chunks[0].Text, "done, here's the result") {
		t.Fatalf("should gather trailing assistant text, got %q", chunks[0].Text)
	}
	if strings.Contains(chunks[0].Text, "first I'll look") {
		t.Fatalf("tool_use turns should be skipped, got %q", chunks[0].Text)
	}
}

func TestChunkWindowsOversizedUnit(t *testing.T) {
	big := strings.Repeat("a", 5000)
	msgs := []types.Message{
		msg("user", big, false, false),
		msg("assistant", "ok", false, false),
	}
	chunks := ChunkMessages("s", msgs, ChunkOptions{MaxChars: 1000, Overlap: 100})
	if len(chunks) < 5 {
		t.Fatalf("5000+ chars at 1000/window should produce >=5 chunks, got %d", len(chunks))
	}
	for _, c := range chunks {
		if len([]rune(c.Text)) > 1000 {
			t.Fatalf("window exceeded MaxChars: %d", len([]rune(c.Text)))
		}
	}
}

func TestWindowOverlap(t *testing.T) {
	s := strings.Repeat("x", 250)
	w := window(s, 100, 20)
	// step = 80; windows start at 0,80,160 — the 160 window reaches end (250)
	// and breaks, covering [0:100][80:180][160:250] -> 3 windows, full coverage.
	if len(w) != 3 {
		t.Fatalf("want 3 overlapping windows, got %d", len(w))
	}
}
