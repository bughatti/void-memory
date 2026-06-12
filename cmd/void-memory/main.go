// Command void-memory is the MCP server entrypoint.
//
// Speaks Model Context Protocol over stdio. Exposes four tools to Claude
// Code: recall, read_session, list_topics, index_status.
//
// Configuration via env vars (all optional):
//   VOID_MEMORY_DATA_DIR     default: ~/.void-memory
//   VOID_MEMORY_PROJECTS_DIR default: ~/.claude/projects
//   VOID_MEMORY_OLLAMA_URL   default: http://localhost:11434
//   VOID_MEMORY_MODEL        default: qwen2.5-coder:7b
//
// Install into Claude Code via ~/.claude.json:
//   {"mcpServers": {"void-memory": {"command": "void-memory.exe"}}}
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/bughatti/void-memory/internal/backend"
	vmi "github.com/bughatti/void-memory/internal/install"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

const (
	appName    = "void-memory"
	appVersion = "0.1.0-alpha"
)

func main() {
	log.SetOutput(os.Stderr)
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.Printf("%s %s starting", appName, appVersion)

	// `install` and `uninstall` run before any backend init — they're
	// bootstrap/teardown steps and must not require an existing Ollama /
	// catalog / projects dir.
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "install":
			runInstallCmd()
			return
		case "uninstall":
			wipe := false
			for _, a := range os.Args[2:] {
				if a == "--wipe" || a == "-wipe" {
					wipe = true
				}
			}
			if err := vmi.Uninstall(wipe); err != nil {
				log.Fatalf("uninstall: %v", err)
			}
			return
		}
	}

	cfg := loadConfig()
	// CLI subcommands run synchronously and would race against the
	// fsnotify-driven background indexer (both processing the same files,
	// doubling LLM cost). Disable the background indexer for one-shot CLI
	// runs; only the MCP serving mode keeps it on.
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "index", "recall", "status", "topics", "reindex", "search":
			cfg.DisableBackgroundIdx = true
		}
	}
	be, err := backend.NewLocal(cfg)
	if err != nil {
		log.Fatalf("local backend init: %v", err)
	}
	defer be.Close()

	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "index":
			runIndexCmd(be)
			return
		case "recall":
			if len(os.Args) < 3 {
				log.Fatal("usage: void-memory recall \"<query>\"")
			}
			runRecallCmd(be, strings.Join(os.Args[2:], " "))
			return
		case "status":
			runStatusCmd(be)
			return
		case "topics":
			cat := ""
			if len(os.Args) > 2 {
				cat = os.Args[2]
			}
			runTopicsCmd(be, cat)
			return
		case "reindex":
			runReindexCmd(be)
			return
		case "search":
			if len(os.Args) < 3 {
				log.Fatal("usage: void-memory search \"<query>\"")
			}
			runSearchCmd(be, strings.Join(os.Args[2:], " "))
			return
		}
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-sig
		log.Printf("signal received, shutting down")
		_ = be.Close()
		os.Exit(0)
	}()

	srv := server.NewMCPServer(appName, appVersion)
	registerTools(srv, be)

	log.Printf("ready: model=%s data_dir=%s projects_dir=%s",
		cfg.Model, cfg.DataDir, cfg.ProjectsDir)

	if err := server.ServeStdio(srv); err != nil {
		log.Fatalf("mcp serve: %v", err)
	}
}

func runInstallCmd() {
	bin, err := os.Executable()
	if err != nil {
		log.Fatalf("locate binary: %v", err)
	}
	abs, _ := filepath.Abs(bin)
	if err := vmi.Run(context.Background(), abs, true); err != nil {
		log.Fatalf("install: %v", err)
	}
}

func runIndexCmd(be *backend.LocalBackend) {
	ctx := context.Background()
	log.Printf("running synchronous index pass...")
	if err := be.IndexNow(ctx); err != nil {
		log.Fatalf("index: %v", err)
	}
	st, _ := be.IndexStatus(ctx)
	b, _ := json.MarshalIndent(st, "", "  ")
	fmt.Println(string(b))
}

func runRecallCmd(be *backend.LocalBackend, query string) {
	ctx := context.Background()
	res, err := be.Recall(ctx, query, backend.RecallHints{MaxTokens: 1500})
	if err != nil {
		log.Fatalf("recall: %v", err)
	}
	b, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(b))
}

func runSearchCmd(be *backend.LocalBackend, query string) {
	ctx := context.Background()
	results, err := be.HybridSearch(ctx, query, 8)
	if err != nil {
		log.Fatalf("search: %v", err)
	}
	fmt.Printf("query: %q\n%d results (retrieval-only, no LLM, CPU):\n", query, len(results))
	for i, r := range results {
		snippet := r.Chunk.Text
		if len(snippet) > 220 {
			snippet = snippet[:220]
		}
		snippet = strings.ReplaceAll(snippet, "\n", " ")
		fmt.Printf("\n#%d  score=%.5f  [%s]\n  %s...\n", i+1, r.Score, r.Chunk.ID, snippet)
	}
}

func runReindexCmd(be *backend.LocalBackend) {
	ctx := context.Background()
	log.Printf("rebuilding hybrid index (chunk + embed all transcripts)...")
	if err := be.ReindexHybrid(ctx); err != nil {
		log.Fatalf("reindex: %v", err)
	}
	log.Printf("hybrid index rebuilt and saved")
}

func runStatusCmd(be *backend.LocalBackend) {
	ctx := context.Background()
	st, err := be.IndexStatus(ctx)
	if err != nil {
		log.Fatalf("status: %v", err)
	}
	b, _ := json.MarshalIndent(st, "", "  ")
	fmt.Println(string(b))
}

func runTopicsCmd(be *backend.LocalBackend, cat string) {
	ctx := context.Background()
	topics, err := be.ListTopics(ctx, cat)
	if err != nil {
		log.Fatalf("topics: %v", err)
	}
	b, _ := json.MarshalIndent(topics, "", "  ")
	fmt.Println(string(b))
}

func loadConfig() backend.LocalConfig {
	home, err := os.UserHomeDir()
	if err != nil {
		log.Fatalf("home dir: %v", err)
	}
	cfg := backend.LocalConfig{
		DataDir:     filepath.Join(home, ".void-memory"),
		ProjectsDir: filepath.Join(home, ".claude", "projects"),
		OllamaURL:   "",
		Model:       "qwen2.5-coder:7b",
	}
	if v := os.Getenv("VOID_MEMORY_DATA_DIR"); v != "" {
		cfg.DataDir = v
	}
	if v := os.Getenv("VOID_MEMORY_PROJECTS_DIR"); v != "" {
		cfg.ProjectsDir = v
	}
	if v := os.Getenv("VOID_MEMORY_OLLAMA_URL"); v != "" {
		cfg.OllamaURL = v
	}
	if v := os.Getenv("VOID_MEMORY_MODEL"); v != "" {
		cfg.Model = v
	}
	if v := os.Getenv("VOID_MEMORY_RETRIEVAL"); v != "" {
		cfg.Retrieval = v
	}
	if v := os.Getenv("VOID_MEMORY_EMBED_MODEL"); v != "" {
		cfg.EmbModel = v
	}
	// Sensible default embed model when hybrid is enabled but unspecified.
	if cfg.Retrieval == "hybrid" && cfg.EmbModel == "" {
		cfg.EmbModel = "nomic-embed-text"
	}
	return cfg
}

// readOnlyTool is the common annotation set for every void-memory tool:
// they all read local transcripts + call a local LLM. None of them
// modify Claude's working tree, run shell commands, hit external APIs,
// or do anything Claude Code's permission system should flag.
//
// These annotations are the canonical signal for "this tool is safe to
// auto-approve" — Claude Code uses them to decide whether to prompt the
// user. Setting them correctly here means users don't have to add allow
// rules in settings.json manually.
func readOnlyToolOpts(extras ...mcp.ToolOption) []mcp.ToolOption {
	base := []mcp.ToolOption{
		mcp.WithReadOnlyHintAnnotation(true),
		mcp.WithDestructiveHintAnnotation(false),
		mcp.WithOpenWorldHintAnnotation(false),
	}
	return append(base, extras...)
}

func registerTools(srv *server.MCPServer, be *backend.LocalBackend) {
	srv.AddTool(
		mcp.NewTool("recall",
			readOnlyToolOpts(
				mcp.WithDescription(
					"REQUIRED: call this at the start of any user prompt that references a "+
						"person, project, character, file, or domain entity. Returns a focused "+
						"<prior-work> block synthesized from prior conversation transcripts. "+
						"Costs you nothing — the synthesis runs on a local LLM, not Claude. "+
						"Default to calling it; only skip on standalone questions ('what is X', "+
						"'list files'). The returned text should be treated as authoritative "+
						"context about what was decided in earlier sessions. SAFE: read-only, "+
						"local-only, no destructive side effects."),
				mcp.WithIdempotentHintAnnotation(false), // LLM synthesis varies between calls
				mcp.WithString("query",
					mcp.Required(),
					mcp.Description("The user's current prompt, verbatim. Don't paraphrase.")),
				mcp.WithString("recent_entities",
					mcp.Description("Optional comma-separated list of entities mentioned in the last few turns of THIS session — improves routing.")),
			)...,
		),
		recallHandler(be),
	)
	srv.AddTool(
		mcp.NewTool("read_session",
			readOnlyToolOpts(
				mcp.WithDescription(
					"Fetch the full transcript of a specific prior session by ID. Use this "+
						"only when the recall synthesis isn't enough and you need to see the "+
						"actual conversation verbatim. Session IDs are cited inline in recall "+
						"outputs. SAFE: read-only, local-only."),
				mcp.WithIdempotentHintAnnotation(true),
				mcp.WithString("session_id",
					mcp.Required(),
					mcp.Description("The session_id returned by a prior recall call.")),
			)...,
		),
		readSessionHandler(be),
	)
	srv.AddTool(
		mcp.NewTool("list_topics",
			readOnlyToolOpts(
				mcp.WithDescription(
					"List indexed sessions with their categories and entity tags. Useful "+
						"for the user to browse what void-memory knows about; not typically "+
						"useful as part of answering a prompt. SAFE: read-only, local-only."),
				mcp.WithIdempotentHintAnnotation(true),
				mcp.WithString("category",
					mcp.Description("Optional category filter: addons, wow-api, voidscout-backend, gameplay, infra, tooling. Empty returns all.")),
			)...,
		),
		listTopicsHandler(be),
	)
	srv.AddTool(
		mcp.NewTool("index_status",
			readOnlyToolOpts(
				mcp.WithDescription(
					"Returns runtime diagnostics: session count indexed, queue depth, "+
						"model in use, whether Ollama is reachable. For debugging only. "+
						"SAFE: read-only, local-only."),
				mcp.WithIdempotentHintAnnotation(true),
			)...,
		),
		indexStatusHandler(be),
	)
}

func recallHandler(be *backend.LocalBackend) server.ToolHandlerFunc {
	return func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		query := req.GetString("query", "")
		if query == "" {
			return mcp.NewToolResultError("query is required"), nil
		}
		hints := backend.RecallHints{MaxTokens: 1500}
		if v := req.GetString("recent_entities", ""); v != "" {
			hints.RecentEntities = splitCSV(v)
		}
		res, err := be.Recall(ctx, query, hints)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("recall: %v", err)), nil
		}
		if res.Synthesis == "" {
			return mcp.NewToolResultText("<prior-work>no relevant prior context found</prior-work>"), nil
		}
		return mcp.NewToolResultText(res.Synthesis), nil
	}
}

func readSessionHandler(be *backend.LocalBackend) server.ToolHandlerFunc {
	return func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		id := req.GetString("session_id", "")
		if id == "" {
			return mcp.NewToolResultError("session_id is required"), nil
		}
		tr, err := be.ReadSession(ctx, id)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("read: %v", err)), nil
		}
		// Truncate the dump — full transcripts can be hundreds of MB.
		const maxChars = 200000
		var out string
		used := 0
		for _, m := range tr.Messages {
			if m.Type != "user" && m.Type != "assistant" {
				continue
			}
			role := m.Type
			line := fmt.Sprintf("%s [%s]: %s\n", role, m.Timestamp.Format("2006-01-02T15:04Z"), m.Content)
			if used+len(line) > maxChars {
				out += "\n[truncated — session is larger than 200KB output cap]"
				break
			}
			out += line
			used += len(line)
		}
		return mcp.NewToolResultText(out), nil
	}
}

func listTopicsHandler(be *backend.LocalBackend) server.ToolHandlerFunc {
	return func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		cat := req.GetString("category", "")
		topics, err := be.ListTopics(ctx, cat)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("list: %v", err)), nil
		}
		b, _ := json.MarshalIndent(topics, "", "  ")
		return mcp.NewToolResultText(string(b)), nil
	}
}

func indexStatusHandler(be *backend.LocalBackend) server.ToolHandlerFunc {
	return func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		st, err := be.IndexStatus(ctx)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("status: %v", err)), nil
		}
		b, _ := json.MarshalIndent(st, "", "  ")
		return mcp.NewToolResultText(string(b)), nil
	}
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
