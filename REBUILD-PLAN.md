# void-memory — Hybrid Retrieval Rebuild Plan

> Branch: `hybrid-retrieval`. Started 2026-06-11. This doc is the canonical, version-controlled
> record of the rebuild so any session/machine can resume exactly where we left off.

## Why we're doing this

The current retrieval path is **LLM-heavy and GPU-bound**: it routes with an LLM, ranks whole
sessions by pre-extracted entity overlap, greps excerpts, then synthesizes with an LLM. On the
primary box (RTX 4060, 8 GB, usually running WoW) the synthesis LLM **fights the game for the
GPU** — 7B thrashed VRAM (47–60 s hangs); even 3B stalls ~1/3 of calls when WoW is foreground.

A deep-research pass (23 sources, 24/25 claims adversarially verified) concluded the win is in the
**index/structure, not a bigger engine** — the "Supercluster move." Key verified findings:

- **Binary embedding quantization**: float32 → 1-bit = 32× smaller, ~25× faster CPU search
  (Hamming = XOR+POPCNT, ~2 cycles), **~96% accuracy retained** with a float rescore. → vector
  search runs on **CPU/NAS, no GPU**.
- **Hybrid lexical + semantic**: BM25/full-text half gives **exact-match, zero-hallucination**
  recall on load-bearing tokens (spell IDs, `C_*` API names, "HARD RULE" keywords) that embeddings
  blur; vector half catches meaning. Fuse with RRF.
- **Defer the LLM to query time** over a small, precisely-retrieved, citation-grounded set →
  minimal hallucination, and the LLM is no longer the bottleneck.
- Hierarchical summary trees (RAPTOR) are the literal Supercluster-analog but are **update-
  sensitive** (full rebuilds) and add ~4% summary hallucination → **deferred to a later optional
  layer**, not in v1. (LazyGraphRAG's LLM-free structure preferred if/when we add it.)
- **Refuted (0-3):** semantic chunking reliably beats fixed-token chunking — do NOT assume it.

## Architecture: dual-mode, user-selectable (local OR server)

`MemoryBackend` (existing interface) is the seam. **Both backends are first-class**, chosen by
config (`VOID_MEMORY_BACKEND=local|server`, default `local`). The hybrid-retrieval **core is
shared** — only where the index lives differs.

```
            ┌──────────────────────────────────────────────┐
            │  Shared hybrid-retrieval core (internal/retrieval) │
            │  chunk · BM25 · binary vectors · RRF · defer LLM    │
            └───────────────────┬──────────────────────────┘
              VOID_MEMORY_BACKEND=local │ =server
            ┌──────────────────────┐  ┌─────────────────────────────┐
            │ LocalBackend (default)│  │ ServerBackend (opt-in)       │
            │ on-device packed files│  │ thin HTTP client → NAS        │
            │ + local Ollama embed  │  │   memory-api (FastAPI)        │
            │ zero network, private │  │   → Postgres + pgvector       │
            └──────────────────────┘  │   aggregates 20+ sessions     │
                                       └─────────────────────────────┘
```

- **Local** = solo user, fully private, no server needed. Dependency-free Go: in-memory inverted
  index (BM25) + packed binary vectors persisted under `~/.void-memory/`, embeddings via a small
  CPU Ollama model. Keeps the project's "JSON files, no SQL" ethos.
- **Server** = pool many machines. NAS `void-memory-db` (pgvector/pg16, **isolated** from the
  existing `voidscout-db`) + a FastAPI `memory-api` (reuses the user's existing NAS stack pattern,
  needs no Go on the NAS). Each machine's MCP becomes a thin client that ships chunks up / queries
  recall down.

### Shared retrieval pipeline (both backends)
1. **Chunk** transcripts into decision-units (user prompt + following assistant text), tagged with
   session id, timestamp, category. Fixed-size with overlap (NOT semantic — that was refuted).
2. **Lexical index**: BM25 over chunks (local: in-memory inverted index; server: Postgres
   tsvector/GIN).
3. **Vector index**: small CPU embedding → binary-quantized (1 bit/dim) + retained float for
   rescore (local: packed bytes + math/bits Hamming; server: pgvector `bit` + Hamming).
4. **Retrieve**: BM25 top-N ∪ binary-Hamming top-N → **RRF fuse** → float rescore top candidates.
5. **(optional) rerank**: small cross-encoder (later).
6. **Synthesize**: existing LLM synthesizer, fed ONLY the top-K fused chunks. Existing faithfulness
   system prompt (cite-or-omit, no-hallucination) is kept.

## Staged execution (each stage independently safe; old path runs until new one is proven)

- [x] **Stage 0 — Safety.** Branch `hybrid-retrieval`. Backup of `~/.void-memory` →
  `C:\Users\liquidai\void-memory-index-backup` (14 meta files). Transcripts
  (`~/.claude/projects/*.jsonl`) confirmed sacred/read-only — never modified. Go toolchain install
  (was missing on this box AND the NAS).
- [~] **Stage 1 — Local hybrid core.** `ollama.Embeddings()`; `internal/retrieval` (chunk, BM25,
  binary-quant + Hamming, RRF) with unit tests; wire `LocalBackend.Recall` to use it (behind a flag
  so the old path is reversible). Validate recall quality vs current 3b on real queries.
  - DONE: retrieval core + tests, ingestion, wiring, CPU-pinned embed, first reindex (~4400 chunks).
  - DONE (cut 1): legacy LLM indexer disabled in hybrid mode (no longer loads the synth LLM on GPU).
  - TODO (cut 2, after validation): DELETE the legacy path entirely — `recall/router.go`,
    `indexer/extract.go` + the watcher's LLM calls, `recall/synthesize.go` (legacy session-scoping
    synthesizer, superseded by `synthesize_chunks.go`), and the LLM-extracted catalog metadata
    (Category/Entities/SubTopics/Summary/Tags). Keep `indexer/parser.go` + `EnumerateProjectsDir`
    (hybrid needs them). Rebuild `list_topics`/`index_status` off the hybrid index.
  - TODO: run hybrid-vs-3b recall comparison on real queries.
- [ ] **Stage 2 — Server store.** Stand up isolated `void-memory-db` (pgvector/pg16) on the NAS.
  Schema: chunks + tsvector + binary vector. (Needs explicit go-ahead — touches production NAS.)
- [ ] **Stage 3 — memory-api (FastAPI).** Ingest/chunk/embed + hybrid retrieve endpoints on the NAS.
- [ ] **Stage 4 — ServerBackend (Go).** Thin HTTP client implementing `MemoryBackend`; config
  selects local vs server. Local fallback if server unreachable.
- [ ] **Stage 5 — Rollout.** Backfill 20+ machines' transcripts; ongoing ingestion; docs.

## Hard constraints / don't-lose rules
- **Reindex embeddings run CPU-pinned (`EmbeddingsCPU`, num_gpu=0).** A GPU batch embed
  job during WoW oversubscribed the 8GB RTX 4060 (WoW + 3B synth model + embed model > 8GB)
  and crashed WoW + the terminal on 2026-06-11. Batch builds have no latency requirement, so
  they belong on the CPU; never run a sustained GPU job alongside the game on this box.
- **Never modify `~/.claude/projects/*.jsonl`** — Claude Code's own transcripts, the source of truth.
- Keep the **old retrieval path working** until the new one is validated; gate the new path behind
  config so rollback is `git checkout main` + restart.
- `void-memory-db` must be **isolated** from `voidscout-db` (separate container/volume/creds).
- Embedding model must be **validated on the user's own transcripts** — the ~96% binary retention is
  model-dependent (one model dropped to ~75%).

## Open questions (from research, to resolve during build)
- Which CPU embedding model survives binary quantization best on coding/gaming transcripts
  (gte-small vs bge-small vs nomic-embed)?
- Best RRF weighting (lexical vs vector) for guaranteeing exact-match on spell IDs / API names.
- Update-cheap hierarchical layer later: periodic RAPTOR re-tree vs LazyGraphRAG vs none.
