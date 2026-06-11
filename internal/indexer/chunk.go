// chunk.go splits long sessions into time-windowed sub-sessions so each
// chunk gets its own category + entity extraction. Solves the case where
// a 10-week session spans wildly different topics and a single category
// assignment loses signal.
//
// Trigger: session span > ChunkThresholdDays OR prompt count > ChunkThresholdPrompts.
// Default chunk window: ChunkWindowDays.
//
// Each chunk inherits JSONLPath from the source so synthesize.go can still
// open the original transcript when reading excerpts — it just filters by
// timestamp to limit reads to the chunk's window.
package indexer

import (
	"fmt"
	"time"

	"github.com/bughatti/void-memory/pkg/types"
)

const (
	ChunkThresholdDays    = 7
	ChunkThresholdPrompts = 500
	ChunkWindowDays       = 7
)

// SessionChunk is a slice of a ParsedSession bounded by [WindowStart, WindowEnd].
// The Extractor processes each chunk independently.
type SessionChunk struct {
	ParentSessionID string
	ChunkIndex      int
	WindowStart     time.Time
	WindowEnd       time.Time
	Messages        []types.Message
	JSONLPath       string
	SourceMTime     time.Time
}

// AsParsedSession synthesizes a ParsedSession view scoped to the chunk
// window, so the existing Extractor code can run on it unchanged.
func (c SessionChunk) AsParsedSession() *ParsedSession {
	return &ParsedSession{
		SessionID:   fmt.Sprintf("%s-chunk-%02d", c.ParentSessionID, c.ChunkIndex),
		Path:        c.JSONLPath,
		StartTime:   c.WindowStart,
		EndTime:     c.WindowEnd,
		SourceMTime: c.SourceMTime,
		Messages:    c.Messages,
	}
}

// ShouldChunk returns true if a session is large enough to warrant
// splitting. Small sessions stay as a single metadata record.
func ShouldChunk(ps *ParsedSession) bool {
	prompts := ps.UserPrompts()
	if len(prompts) > ChunkThresholdPrompts {
		return true
	}
	if !ps.StartTime.IsZero() && !ps.EndTime.IsZero() {
		span := ps.EndTime.Sub(ps.StartTime)
		if span > time.Duration(ChunkThresholdDays)*24*time.Hour {
			return true
		}
	}
	return false
}

// ChunkSession splits a parsed session into time-windowed chunks. Each
// chunk contains the original messages whose timestamps fall in the
// window. The last chunk extends through ps.EndTime even if its window
// would technically end earlier.
func ChunkSession(ps *ParsedSession) []SessionChunk {
	window := time.Duration(ChunkWindowDays) * 24 * time.Hour
	if ps.StartTime.IsZero() || ps.EndTime.IsZero() {
		return []SessionChunk{singleChunk(ps)}
	}

	var chunks []SessionChunk
	idx := 0
	cursor := ps.StartTime
	for cursor.Before(ps.EndTime) {
		end := cursor.Add(window)
		if end.After(ps.EndTime) {
			end = ps.EndTime
		}
		// Inclusive lower bound, exclusive upper bound (except for the last
		// chunk where upper is inclusive of EndTime to capture trailing
		// messages).
		isLast := !end.Before(ps.EndTime)
		var chunkMsgs []types.Message
		for _, m := range ps.Messages {
			if m.Timestamp.Before(cursor) {
				continue
			}
			if isLast {
				if m.Timestamp.After(end) {
					continue
				}
			} else {
				if !m.Timestamp.Before(end) {
					continue
				}
			}
			chunkMsgs = append(chunkMsgs, m)
		}
		// Skip empty chunks — they happen when the user idles for >1 week.
		if hasUserPrompt(chunkMsgs) {
			chunks = append(chunks, SessionChunk{
				ParentSessionID: ps.SessionID,
				ChunkIndex:      idx,
				WindowStart:     cursor,
				WindowEnd:       end,
				Messages:        chunkMsgs,
				JSONLPath:       ps.Path,
				SourceMTime:     ps.SourceMTime,
			})
			idx++
		}
		cursor = end
	}
	if len(chunks) == 0 {
		return []SessionChunk{singleChunk(ps)}
	}
	return chunks
}

func singleChunk(ps *ParsedSession) SessionChunk {
	return SessionChunk{
		ParentSessionID: ps.SessionID,
		ChunkIndex:      0,
		WindowStart:     ps.StartTime,
		WindowEnd:       ps.EndTime,
		Messages:        ps.Messages,
		JSONLPath:       ps.Path,
		SourceMTime:     ps.SourceMTime,
	}
}

func hasUserPrompt(msgs []types.Message) bool {
	for _, m := range msgs {
		if m.Type == "user" && !m.IsToolResult {
			return true
		}
	}
	return false
}
