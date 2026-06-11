// Package taxonomy encodes the v1 category + entity taxonomy derived from
// transcript analysis. See TAXONOMY-v1.md for the research that produced this.
// The indexer prompts the local LLM with these definitions to keep
// classification consistent across sessions.
package taxonomy

import "github.com/bughatti/void-memory/pkg/types"

// CategoryDef is the canonical definition of one category. The LLM
// classifier gets this struct in its prompt so it knows what each bucket
// means and what the user-friendly edge-case rules are.
type CategoryDef struct {
	ID          types.Category
	Description string
	SubTopics   []string
	Examples    []string
}

// Definitions is the v1 taxonomy. Sourced from parallel agent analysis of
// 5173 user prompts across 2 main session JSONLs (2026-03-13 → 2026-06-11).
var Definitions = []CategoryDef{
	{
		ID: types.CategoryAddons,
		Description: "WoW Lua addon code. UI frames, event handlers, slash commands, " +
			"SecureActionButtons, module implementation, in-game-side feature work. " +
			"Anything that ships as a wow-addons/<X>/*.lua file.",
		SubTopics: []string{"module-implementation", "ui-layout", "bug-fix", "cross-addon-sync"},
		Examples: []string{
			"build a tank-swap popup module for Voidspire",
			"the KickRotation panel still says up next Vede when it shouldn't",
			"fix VoidBags click-to-use breaking",
		},
	},
	{
		ID: types.CategoryWowAPI,
		Description: "Research / understanding of Blizzard's API, taint system, " +
			"secret values. UNDERSTANDING-level, not implementation-level. The " +
			"output becomes authoritative-reference material.",
		SubTopics: []string{"taint-and-secrets", "api-survey", "workaround-discovery", "community-addon-mining"},
		Examples: []string{
			"how does ETEA work and what fields are safe",
			"why does CLEU taint everything in 12.0.5",
			"survey C_DamageMeter return values",
		},
	},
	{
		ID: types.CategoryVoidScoutBackend,
		Description: "VoidScout-the-pipeline: FastAPI on NAS, Postgres schemas, Go " +
			"uploader, voidscout.io web frontend, scoring algorithm math. NOT the " +
			"in-game addon code (that's `addons`).",
		SubTopics: []string{"scoring-math", "api-endpoint", "web-frontend", "data-pipeline", "privacy-and-opt-out"},
		Examples: []string{
			"add Bayesian shrinkage to guild util scores",
			"voidscout.io smoke test failing on /api/character/littlevede",
			"build /raid-probability page",
		},
	},
	{
		ID: types.CategoryGameplay,
		Description: "WoW play decisions and analysis. Vault choices, stat priorities, " +
			"rotation questions, raid kill strategy, M+ comp choices, character " +
			"progression. NOT addon code — actual play questions.",
		SubTopics: []string{"gear-and-stats", "rotation", "raid-mechanics", "m-plus", "multi-char"},
		Examples: []string{
			"vault gives me a myth ring vs hero ring - which",
			"i got Liferipper's Cutlass mythic, swap from crafted Warblade?",
			"dont I need taunt out on first boss at all times",
		},
	},
	{
		ID: types.CategoryInfra,
		Description: "NAS, Docker, networks, deployments, host OS, CI/CD. Below " +
			"the application layer. claude-mem stack, voidscout container topology, " +
			"cloudflared, Postgres setup, GitHub Actions, Windows host scripting.",
		SubTopics: []string{"nas-docker", "deploy-and-ci", "host-setup", "network-topology"},
		Examples: []string{
			"set up cloudflared for voidscout.io",
			"GitHub Actions deploy is broken — token IP filter",
			"NAS is at 192.168.1.10, docker isolation pattern",
		},
	},
	{
		ID: types.CategoryTooling,
		Description: "Meta: tools that improve the dev workflow itself. claude-mem, " +
			"MCP servers, void-memory (this project), Claude Code config, slash " +
			"commands, voice transcript pipeline, code-review skills.",
		SubTopics: []string{"claude-tools", "dev-workflow", "memory-system"},
		Examples: []string{
			"deploy claude-mem on NAS",
			"set up VoidVoice for voice prompts",
			"build void-memory MCP server",
		},
	},
}

// EdgeCaseRules captures the priority rules when a prompt could fit two
// categories. The LLM gets these verbatim in its routing prompt.
const EdgeCaseRules = `
When a prompt could fit two categories, use this priority:
  - addons over wow-api: if it's about implementing in a specific addon
  - wow-api over addons: if it's about understanding Blizzard's API generically
  - addons over gameplay: if it's about coding gear/rotation features
  - voidscout-backend over addons: for the VoidScout backend pipeline; use addons for in-game VoidScout
  - infra over voidscout-backend: if it's container/network/Docker concerns
`

// SeedEntities is the manually-curated starting entity catalog. The indexer
// is allowed to add to this set as it processes sessions — these are just
// the known knowns from research.
var SeedEntities = struct {
	Characters     []string
	Projects       []string
	Subsystems     []string
	Concepts       []string
	Bosses         []string
	Raids          []string
	MPlusDungeons  []string
	Hosts          []string
	ExternalServices []string
}{
	Characters: []string{
		"Vede", "Vede-tank", "Khiwi", "Jish-Area52",
		"Sannesh-Proudmoore", "Kaidaa", "Tulao-Feathermoon-US",
	},
	Projects: []string{
		"VoidScout", "VRT", "VRTReader", "VoidUI", "VoidBags",
		"VoidFisher", "VoidCheatSheet", "VoidCalendar", "VoidHub",
		"VoidLFG", "VoidPug", "VoidWatcher", "VoidDice", "VoidAlert",
		"VoidLib", "voidscout-data", "voidscout-uploader", "voidscout-web",
		"voidscout.io", "void-memory", "claude-mem",
	},
	Subsystems: []string{
		"KickRotation", "FightRecorder", "ScoreEngine", "TankSwap",
		"Lura", "BossAlertEngine", "MarkerScan", "TrashDiscovery",
		"LFGPanel", "CharacterAudit", "EncounterJournal", "RareTracker",
		"MythicPlus.lua", "BossMods.lua", "mechanics_lib.py", "guild_endpoint.py",
	},
	Concepts: []string{
		"CLEU", "taint", "secret-values", "ETEA", "C_DamageMeter",
		"C_UnitAuras", "C_EncounterTimeline", "C_LFGList",
		"SecureActionButton", "kstring", "Voidforge", "Bayesian-shrinkage",
		"8-axis-model", "subscription-OAuth", "MCP", "BullMQ", "Postgres-FTS5",
	},
	Bosses: []string{
		"Chimaerus", "L'ura", "Bellamy", "Averzian", "Crown", "Belo'ren",
		"Midnight-Falls", "Salhadaar", "Vanguard", "Imperator", "Vorasius",
	},
	Raids: []string{
		"Voidspire", "Dreamrift", "March-on-Quel'Danas",
	},
	MPlusDungeons: []string{
		"MT", "Seat", "Skyreach", "Spire", "Maisara",
		"Nexus-Point", "Windrunner-Spire", "Pit-of-Saron",
	},
	Hosts: []string{
		"NAS", "192.168.1.10", "voidscout.io", "Windows-gaming-PC",
		"voidscout-net", "voidscout-db", "voidscout-api",
	},
	ExternalServices: []string{
		"CurseForge", "GitHub", "WCL", "Raider.io", "Carried.io",
		"Archon", "Wago.tools", "Anthropic", "Cloudflare", "Ollama",
	},
}

// AllSeedEntities returns the flat list of all pre-known entities for
// the LLM to match against during extraction.
func AllSeedEntities() []string {
	var out []string
	out = append(out, SeedEntities.Characters...)
	out = append(out, SeedEntities.Projects...)
	out = append(out, SeedEntities.Subsystems...)
	out = append(out, SeedEntities.Concepts...)
	out = append(out, SeedEntities.Bosses...)
	out = append(out, SeedEntities.Raids...)
	out = append(out, SeedEntities.MPlusDungeons...)
	out = append(out, SeedEntities.Hosts...)
	out = append(out, SeedEntities.ExternalServices...)
	return out
}
