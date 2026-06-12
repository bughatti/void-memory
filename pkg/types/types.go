// Package types defines public type definitions shared across void-memory.
// Other packages reference these instead of redefining shapes.
package types

import "time"

// SessionMeta is lightweight per-session info derived from the hybrid index,
// returned by ListTopics for manual exploration. (The old LLM-extracted
// metadata — categories, entities, summaries — was removed with the legacy
// indexer; retrieval is now purely BM25 + vector over chunks.)
type SessionMeta struct {
	SessionID  string    `json:"session_id"`
	StartTime  time.Time `json:"start_time"`
	EndTime    time.Time `json:"end_time"`
	ChunkCount int       `json:"chunk_count"`
}

// RecallResult is what gets returned to Claude via the MCP recall() tool.
type RecallResult struct {
	Synthesis     string    `json:"synthesis"`      // the actual <prior-work> block
	SessionsUsed  []string  `json:"sessions_used"`  // session IDs the chunks came from
	TokenEstimate int       `json:"token_estimate"` // approximate tokens
	GeneratedAt   time.Time `json:"generated_at"`
}

// SessionTranscript is the parsed view of a JSONL session. Loaded lazily — only
// when ReadSession is called for the raw conversation.
type SessionTranscript struct {
	SessionID string
	Path      string
	Messages  []Message
}

// Message is one entry from the JSONL stream — either user or assistant.
type Message struct {
	Type         string    `json:"type"`      // "user", "assistant"
	Timestamp    time.Time `json:"timestamp"`
	Content      string    `json:"content"`   // flattened text content
	IsToolUse    bool      `json:"is_tool_use"`
	IsToolResult bool      `json:"is_tool_result"`
}
