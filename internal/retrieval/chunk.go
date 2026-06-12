// chunk.go turns a session's flat message list into retrievable "decision
// units": each user prompt paired with the assistant's response to it. This is
// the unit of relevance — the question plus what was decided/answered.
//
// Chunking is structural + fixed-window, NOT semantic-boundary: the research
// pass explicitly refuted (0-3) the claim that cosine-boundary semantic
// chunking reliably beats fixed-size chunking, so we don't pay that complexity.
// Oversized units are split into overlapping windows so no content is lost to
// truncation.
package retrieval

import (
	"strings"
	"time"

	"github.com/bughatti/void-memory/pkg/types"
)

// Chunk is one retrievable unit with the metadata needed to cite it back.
type Chunk struct {
	ID        string    // "<sessionID>#<n>"
	SessionID string    //
	Index     int       // ordinal within the session
	Text      string    // "USER: ...\nCLAUDE: ..."
	StartTime time.Time //
	EndTime   time.Time //
}

// ChunkOptions controls windowing. Zero value uses sensible defaults.
type ChunkOptions struct {
	MaxChars int // max chars per chunk before windowing (default 1500)
	Overlap  int // char overlap between windows (default 200)
}

func (o ChunkOptions) withDefaults() ChunkOptions {
	if o.MaxChars <= 0 {
		o.MaxChars = 1500
	}
	if o.Overlap < 0 || o.Overlap >= o.MaxChars {
		o.Overlap = 200
	}
	return o
}

// ChunkMessages builds chunks for one session. msgs is the full ordered message
// list (as produced by indexer.ParseFile). Noise user lines (system reminders,
// task notifications, command tags, tool results) are skipped as prompt anchors
// but assistant text between a prompt and the next real prompt is gathered.
func ChunkMessages(sessionID string, msgs []types.Message, opts ChunkOptions) []Chunk {
	opts = opts.withDefaults()
	var chunks []Chunk
	i := 0
	for i < len(msgs) {
		if !isRealUserPrompt(msgs[i]) {
			i++
			continue
		}
		userText := strings.TrimSpace(msgs[i].Content)
		startT := msgs[i].Timestamp
		endT := startT

		// Gather assistant text turns until the next real user prompt.
		var asst []string
		j := i + 1
		for j < len(msgs) {
			n := msgs[j]
			if isRealUserPrompt(n) {
				break
			}
			if n.Type == "assistant" && !n.IsToolUse {
				if c := strings.TrimSpace(n.Content); c != "" {
					asst = append(asst, c)
				}
			}
			if !n.Timestamp.IsZero() {
				endT = n.Timestamp
			}
			j++
		}

		unit := "USER: " + userText
		if len(asst) > 0 {
			unit += "\nCLAUDE: " + strings.Join(asst, "\n")
		}
		for _, w := range window(unit, opts.MaxChars, opts.Overlap) {
			chunks = append(chunks, Chunk{
				ID:        sessionID + "#" + itoa(len(chunks)),
				SessionID: sessionID,
				Index:     len(chunks),
				Text:      w,
				StartTime: startT,
				EndTime:   endT,
			})
		}
		i = j
	}
	return chunks
}

// isRealUserPrompt reports whether a message is a human-typed prompt (not a
// tool result or one of the Claude-Code-emitted noise lines).
func isRealUserPrompt(m types.Message) bool {
	if m.Type != "user" || m.IsToolResult {
		return false
	}
	s := strings.TrimSpace(m.Content)
	if s == "" {
		return false
	}
	if strings.HasPrefix(s, "<system-reminder>") ||
		strings.HasPrefix(s, "<local-command-stdout>") ||
		strings.HasPrefix(s, "<command-") ||
		strings.HasPrefix(s, "<task-notification>") {
		return false
	}
	return true
}

// window splits s into overlapping windows of at most maxChars. Returns one
// window when s fits. Splits on rune boundaries to avoid cutting UTF-8.
func window(s string, maxChars, overlap int) []string {
	r := []rune(s)
	if len(r) <= maxChars {
		return []string{s}
	}
	var out []string
	step := maxChars - overlap
	for start := 0; start < len(r); start += step {
		end := start + maxChars
		if end > len(r) {
			end = len(r)
		}
		out = append(out, string(r[start:end]))
		if end == len(r) {
			break
		}
	}
	return out
}

// itoa is a tiny non-allocating-ish int->string for chunk ids (avoids importing
// strconv for one call site; keeps the package's import surface minimal).
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	pos := len(b)
	for n > 0 {
		pos--
		b[pos] = byte('0' + n%10)
		n /= 10
	}
	return string(b[pos:])
}
