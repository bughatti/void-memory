// Package types defines public type definitions shared across void-memory.
// Other packages reference these instead of redefining shapes.
package types

import "time"

// Category is the top-level orthogonal bucket for a session. See the v1
// taxonomy in internal/taxonomy for the enumeration. New categories may
// emerge from data; the LLM is allowed to propose them at index time.
type Category string

const (
	CategoryAddons           Category = "addons"
	CategoryWowAPI           Category = "wow-api"
	CategoryVoidScoutBackend Category = "voidscout-backend"
	CategoryGameplay         Category = "gameplay"
	CategoryInfra            Category = "infra"
	CategoryTooling          Category = "tooling"
)

// SessionMeta is the per-session metadata extracted by the indexer and
// queried at recall time. Persisted to ~/.void-memory/sessions/<id>.meta.json.
//
// For long-running sessions (>500 prompts or >7 days), the indexer splits
// the JSONL into time-windowed chunks and emits one SessionMeta per chunk.
// ParentSessionID points back to the original JSONL session_id; SessionID
// for chunks is "<parent>-chunk-<NN>". Single-bucket sessions leave
// ParentSessionID empty and SessionID equal to the JSONL session_id.
type SessionMeta struct {
	SessionID    string    `json:"session_id"`
	ParentSessionID string `json:"parent_session_id,omitempty"`
	ChunkIndex   int       `json:"chunk_index,omitempty"`
	JSONLPath    string    `json:"jsonl_path"`
	StartTime    time.Time `json:"start_time"`
	EndTime      time.Time `json:"end_time"`
	PromptCount  int       `json:"prompt_count"`
	Category     Category  `json:"category"`
	SubTopics    []string  `json:"sub_topics"`
	Entities     []string  `json:"entities"`
	Summary      string    `json:"summary"`       // 1-3 sentence summary
	Tags         []string  `json:"tags"`          // cross-cutting tags (decision-vs-execution, etc.)
	IndexedAt    time.Time `json:"indexed_at"`
	SourceMTime  time.Time `json:"source_mtime"`  // mtime of JSONL when last indexed
	SchemaVer    int       `json:"schema_ver"`    // bump when SessionMeta shape changes
}

// Catalog is the in-memory view loaded at startup. Backed by per-session
// .meta.json files on disk.
type Catalog struct {
	Sessions     []SessionMeta `json:"sessions"`
	KnownEntities []string     `json:"known_entities"` // dedupe / autocomplete
	LastRefresh  time.Time     `json:"last_refresh"`
}

// RouteResult is what the local LLM produces during the routing pass.
// Drives which sessions get loaded for synthesis.
type RouteResult struct {
	NeedsContext bool     `json:"needs_context"` // skip retrieval entirely if false
	Category     Category `json:"category"`
	Entities     []string `json:"entities"`
	Keywords     []string `json:"keywords"`
	Reasoning    string   `json:"reasoning"`     // for debugging / introspection
}

// RecallResult is what gets returned to Claude via the MCP recall() tool.
type RecallResult struct {
	Synthesis      string        `json:"synthesis"`        // the actual context block
	SessionsUsed   []string      `json:"sessions_used"`    // session IDs cited
	TokenEstimate  int           `json:"token_estimate"`   // approximate tokens
	RoutingResult  RouteResult   `json:"routing_result"`   // for transparency
	GeneratedAt    time.Time     `json:"generated_at"`
}

// SessionTranscript is the parsed view of a JSONL session. We don't load
// these eagerly — only when synthesis needs the actual content.
type SessionTranscript struct {
	SessionID string
	Path      string
	Messages  []Message
}

// Message is one entry from the JSONL stream — either user, assistant, or system.
type Message struct {
	Type      string    `json:"type"`      // "user", "assistant", "system", etc.
	Timestamp time.Time `json:"timestamp"`
	Content   string    `json:"content"`   // flattened text content
	IsToolUse bool      `json:"is_tool_use"`
	IsToolResult bool   `json:"is_tool_result"`
}
