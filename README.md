# void-memory

**Persistent session memory for Claude Code. Local-first. Lossless. Zero ongoing API cost.**

> Most "AI memory" tools compress your past conversations into one-line summaries and lose the actual analytical work. void-memory doesn't. It indexes your raw Claude Code transcripts and uses a local LLM (Ollama) to surface relevant prior work when a topic comes up — automatically, without you having to type "remember this."

## The problem this solves

You spend three days analyzing your character's gear in Claude Code. Two weeks later you ask "I got a new weapon, what do I do?" and Claude has zero memory of the prior work. Existing memory tools (ChatGPT Memory, Cursor memory, claude-mem, etc.) compress that conversation into one-liners like *"discussed gear"* and lose the BiS math, the rotation rationale, the stat priorities.

void-memory takes a different bet: **raw transcripts are the source of truth.** They're already on disk at `~/.claude/projects/*.jsonl`. A local LLM reads them on demand, finds what's relevant, synthesizes a focused recall block, and hands it to Claude.

- You don't pay Anthropic for memory. The local LLM is free.
- Claude gets full prior context — the actual decisions, not lossy summaries.
- It's "AI all the way down" — not keyword search dressed up as memory.

## Architecture

```
~/.claude/projects/*.jsonl    ← Claude Code's own transcript files (already on disk)
        ↓ (fsnotify watcher)
Indexer
   • Time-windowed chunking for long sessions (weeks → weekly buckets)
   • Deterministic entity pre-scan against a seeded catalog
   • Local LLM categorizes + extracts metadata
        ↓
~/.void-memory/sessions/<id>.meta.json    ← per-chunk metadata (JSON files, no SQL)
        ↓
MCP server (Go binary, ~10MB)
   • recall(query) — main tool, called by Claude on every relevant prompt
   • read_session(id) — verbatim fallback
   • list_topics() — browse catalog
   • index_status() — diagnostics
        ↓
Claude Code reads recall output as context
```

## Quick start

```bash
# 1. Install Ollama from https://ollama.com/download
# 2. Build the binary
go build -o void-memory ./cmd/void-memory

# 3. Run the installer — detects hardware, pulls the right model, wires Claude Code
./void-memory install

# 4. Open a new Claude Code session. Done.
```

The `install` command does ALL of this with one invocation:
- Detects your hardware tier (GPU/VRAM/CPU/RAM)
- Pulls the right Ollama model for your tier
- Registers void-memory as an MCP server in `~/.claude.json`
- Adds permission allow-rules so there are no prompts
- Runs an initial index pass over your existing transcripts

## Hardware tiers

void-memory auto-detects and picks the right local model at install time.

| Tier | Hardware | Default model | Synthesis latency |
|---|---|---|---|
| S | 24GB+ VRAM | Qwen 2.5 Coder 32B | ~500ms |
| A | 12-16GB VRAM | Qwen 2.5 Coder 14B | ~1s |
| B | 8-12GB VRAM | Qwen 2.5 Coder 7B | ~1.5s |
| C | CPU only, 16GB+ RAM | Phi-3.5 Mini 3.8B | ~3-5s |
| D | Weaker | Fallback: FTS-only | ~50ms |
| E | Remote | Server backend (Phase 6) | ~200ms LAN |

## CLI reference

```bash
void-memory install              # First-time setup (this is the main one)
void-memory uninstall            # Reverse install; add --wipe to also clear ~/.void-memory
void-memory                      # Run as MCP server (Claude Code launches this automatically)
void-memory index                # Force a synchronous reindex pass
void-memory recall "<query>"     # Test recall from the CLI without going through Claude Code
void-memory status               # Print catalog stats
void-memory topics [<category>]  # Browse indexed sessions
```

## How retrieval works

A query like *"I got a new 2H weapon for Vede, should I swap from my crafted Warblade?"* goes through:

1. **Router** (local LLM): classify into `category=gameplay`, `entities=["Vede", "Warblade"]`, `keywords=["weapon", "swap", "crafted"]`. ~0.7-2.8s.
2. **Scoping**: filter the catalog to gameplay sessions tagged with Vede. Out of 50 sessions, you get 5 candidates instead of all 50.
3. **Excerpts**: open those candidate JSONLs, pull only the user prompts mentioning Vede/Warblade plus the immediately-following assistant turns.
4. **Synthesizer** (local LLM): read the excerpts, produce a focused `<prior-work>` block tuned to the current question. ~10-15s.
5. **Return**: the synthesis goes back to Claude as a tool result. ~400 tokens added to Claude's context.

Net cost to your Claude subscription: about $0.006 per recall (or ~0.001% of your daily Max quota). The entire heavy lift runs on your local hardware.

## Long-session handling

Run a single Claude Code session for 10 weeks? The indexer auto-detects sessions with >500 prompts or >7-day spans and splits them into time-windowed chunks. Each chunk gets its own category + entity extraction. Recall picks the right chunk(s) instead of bucketing the whole session as one topic.

## Privacy

- 100% local. No telemetry. No cloud calls except the model pull from ollama.com.
- Transcripts never leave your machine.
- The recall block IS sent to Anthropic (it's the tool response Claude reads), but that's bounded by the synthesis token cap (~1500 tokens default).

## License

MIT. See [LICENSE](LICENSE).

## Status

Pre-alpha. Built for personal use, opened up. PRs and issues welcome.
