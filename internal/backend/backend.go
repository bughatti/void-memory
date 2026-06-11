// Package backend defines the MemoryBackend interface. Two implementations:
//   - LocalBackend: reads/writes ~/.void-memory/, calls local Ollama
//   - ServerBackend: talks HTTP to a central server (Phase 6)
//
// The MCP server (cmd/void-memory) holds a reference to whichever backend the
// user configured. From the MCP server's perspective, both look identical.
// This is the abstraction that lets server-mode be additive rather than a
// rewrite.
package backend

import (
	"context"

	"github.com/bughatti/void-memory/pkg/types"
)

// MemoryBackend is the contract both LocalBackend and ServerBackend implement.
type MemoryBackend interface {
	// Recall is the primary entrypoint: given a user prompt and conversation
	// hints, return a synthesized recall block of relevant prior context.
	// Implementations decide HOW to retrieve (local LLM + files vs HTTP call
	// to remote server) but the contract is the same.
	Recall(ctx context.Context, query string, hints RecallHints) (*types.RecallResult, error)

	// ReadSession returns the full transcript for a session ID. Used when
	// the synthesis isn't enough and Claude wants the raw conversation.
	ReadSession(ctx context.Context, sessionID string) (*types.SessionTranscript, error)

	// ListTopics returns the catalog of session metadata, optionally filtered
	// by category. Used for manual exploration / debugging.
	ListTopics(ctx context.Context, category string) ([]types.SessionMeta, error)

	// IndexStatus returns runtime telemetry: how many sessions indexed, when
	// last refreshed, queue depth if any, what model is in use.
	IndexStatus(ctx context.Context) (*Status, error)

	// Close releases resources. Called on MCP server shutdown.
	Close() error
}

// RecallHints carries optional context that may improve retrieval quality.
// Caller (the MCP server) is allowed to leave any field zero.
type RecallHints struct {
	WorkingDir      string   // user's cwd, if known
	RecentEntities  []string // entities mentioned in the last few turns
	MaxTokens       int      // cap on synthesis output (default: 1500)
	MinConfidence   float64  // skip retrieval if route confidence below this
}

// Status is implementation-agnostic runtime info.
type Status struct {
	Backend          string `json:"backend"` // "local" or "server"
	SessionCount     int    `json:"session_count"`
	LastIndexedAt    string `json:"last_indexed_at"`
	ModelInUse       string `json:"model_in_use"`
	OllamaReachable  bool   `json:"ollama_reachable"`
	IndexerQueueLen  int    `json:"indexer_queue_len"`
}
