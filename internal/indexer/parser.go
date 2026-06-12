// Package indexer reads and normalizes Claude Code JSONL transcripts. It is now
// purely a transcript parser — the hybrid retrieval pipeline (internal/retrieval
// + internal/backend) consumes ParsedSession/Messages to chunk and embed. The
// old LLM metadata extractor and fsnotify catalog watcher were removed.
package indexer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/bughatti/void-memory/pkg/types"
)

// ParsedSession is the flat view of one JSONL file. Used as input to the
// LLM extraction step and as the source of truth for ReadSession() calls.
type ParsedSession struct {
	SessionID   string
	Path        string
	StartTime   time.Time
	EndTime     time.Time
	SourceMTime time.Time
	Messages    []types.Message
}

// UserPrompts returns just the user-typed text messages, filtering out tool
// results, system reminders, and other Claude-Code-emitted user-type lines.
// These are the topic signal — what the human actually said.
func (p *ParsedSession) UserPrompts() []types.Message {
	out := make([]types.Message, 0, 64)
	for _, m := range p.Messages {
		if m.Type != "user" || m.IsToolResult {
			continue
		}
		s := strings.TrimSpace(m.Content)
		if s == "" {
			continue
		}
		// Skip the noise classes we identified during research: system
		// reminders, local command outputs, command tags, task notifications.
		if strings.HasPrefix(s, "<system-reminder>") ||
			strings.HasPrefix(s, "<local-command-stdout>") ||
			strings.HasPrefix(s, "<command-") ||
			strings.HasPrefix(s, "<task-notification>") {
			continue
		}
		out = append(out, m)
	}
	return out
}

// ParseFile reads one JSONL transcript end-to-end. Returns the flattened
// session with messages ordered by their original line order.
func ParseFile(path string) (*ParsedSession, error) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	sessionID := strings.TrimSuffix(filepath.Base(path), ".jsonl")
	ps := &ParsedSession{
		SessionID:   sessionID,
		Path:        path,
		SourceMTime: st.ModTime(),
		Messages:    make([]types.Message, 0, 256),
	}

	// The JSONL lines can be very long (long assistant outputs with embedded
	// file content); raise the scanner buffer to cope.
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1<<20), 32<<20) // start 1MB, grow to 32MB

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		msg, ok := parseLine(line)
		if !ok {
			continue
		}
		if ps.StartTime.IsZero() || msg.Timestamp.Before(ps.StartTime) {
			if !msg.Timestamp.IsZero() {
				ps.StartTime = msg.Timestamp
			}
		}
		if msg.Timestamp.After(ps.EndTime) {
			ps.EndTime = msg.Timestamp
		}
		ps.Messages = append(ps.Messages, msg)
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		return nil, fmt.Errorf("scan %s: %w", path, err)
	}
	return ps, nil
}

// parseLine extracts a normalized Message from one JSONL line. Returns
// (zero, false) if the line is metadata/snapshot/system that we don't
// want to treat as a conversation message.
func parseLine(line []byte) (types.Message, bool) {
	// We need to look at: type, timestamp, message.content (may be string
	// or array). Use a shallow decode to avoid allocating for fields we
	// don't care about.
	var probe struct {
		Type      string          `json:"type"`
		Timestamp string          `json:"timestamp"`
		Message   json.RawMessage `json:"message"`
	}
	if err := json.Unmarshal(line, &probe); err != nil {
		return types.Message{}, false
	}
	if probe.Type != "user" && probe.Type != "assistant" {
		return types.Message{}, false
	}
	out := types.Message{Type: probe.Type}
	if probe.Timestamp != "" {
		if t, err := time.Parse(time.RFC3339Nano, probe.Timestamp); err == nil {
			out.Timestamp = t
		}
	}
	if len(probe.Message) == 0 {
		return out, true
	}
	// message.content can be: a string OR an array of content blocks.
	var content struct {
		Content json.RawMessage `json:"content"`
	}
	_ = json.Unmarshal(probe.Message, &content)
	if len(content.Content) == 0 {
		return out, true
	}
	// Try string first.
	var asStr string
	if err := json.Unmarshal(content.Content, &asStr); err == nil {
		out.Content = asStr
		return out, true
	}
	// Else array of blocks.
	var blocks []struct {
		Type    string          `json:"type"`
		Text    string          `json:"text"`
		Content json.RawMessage `json:"content"`
		IsError bool            `json:"is_error"`
	}
	if err := json.Unmarshal(content.Content, &blocks); err != nil {
		return out, true
	}
	var texts []string
	for _, b := range blocks {
		switch b.Type {
		case "text":
			texts = append(texts, b.Text)
		case "tool_use":
			out.IsToolUse = true
		case "tool_result":
			out.IsToolResult = true
			// Tool results may have content as string or as nested blocks.
			var rstr string
			if err := json.Unmarshal(b.Content, &rstr); err == nil {
				texts = append(texts, rstr)
			}
		}
	}
	out.Content = strings.Join(texts, "\n")
	return out, true
}

// EnumerateProjectsDir walks ~/.claude/projects/*.jsonl and returns paths
// of all top-level session JSONLs (skipping subagent / workflow noise).
// Top-level files have names like <session-id>.jsonl directly under the
// project dir; subagent JSONLs live under <session-id>/subagents/ subtrees.
func EnumerateProjectsDir(rootDir string) ([]string, error) {
	entries, err := os.ReadDir(rootDir)
	if err != nil {
		return nil, err
	}
	var paths []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		projDir := filepath.Join(rootDir, e.Name())
		ee, err := os.ReadDir(projDir)
		if err != nil {
			continue
		}
		for _, f := range ee {
			if f.IsDir() {
				continue
			}
			if !strings.HasSuffix(f.Name(), ".jsonl") {
				continue
			}
			paths = append(paths, filepath.Join(projDir, f.Name()))
		}
	}
	return paths, nil
}
