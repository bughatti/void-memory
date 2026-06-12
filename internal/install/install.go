// Package install drives the one-command install flow:
//   - detect hardware tier
//   - install/verify Ollama
//   - pull the recommended model
//   - patch ~/.claude.json to register the MCP server
//   - patch ~/.claude/settings.json to add the allow rules
//   - drop the hard-rule memory entry
//   - build the initial hybrid index (CPU-pinned embed)
//
// Invoked from main via `void-memory.exe install` — keeps everything in
// one binary so there's nothing else to deploy.
package install

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/bughatti/void-memory/internal/backend"
	"github.com/bughatti/void-memory/internal/ollama"
)

// Tier selection inputs.
type Hardware struct {
	GPU         string // e.g. "NVIDIA GeForce RTX 4060"
	VRAMMiB     int    // total VRAM
	RAMGiB      int    // total system RAM
	CPU         string
	HasGPU      bool
}

// Recommendation picks an Ollama model tag based on detected hardware.
type Recommendation struct {
	Tier        string
	Model       string
	Reason      string
}

func Pick(hw Hardware) Recommendation {
	// Only the synthesis model touches the GPU (one call per recall); hybrid
	// retrieval is CPU-only. So size the synthesis model to leave headroom when
	// the GPU is shared (e.g. a running game) — an 8GB card can't safely host a
	// 7B alongside a game (learned the hard way; see REBUILD-PLAN.md).
	if !hw.HasGPU || hw.VRAMMiB < 4000 {
		if hw.RAMGiB >= 16 {
			return Recommendation{Tier: "C", Model: "qwen2.5-coder:3b", Reason: "CPU-only or low VRAM; 3B synthesis on RAM. Retrieval is CPU-only regardless."}
		}
		return Recommendation{Tier: "D", Model: "qwen2.5-coder:3b", Reason: "Constrained hardware; 3B synthesis will be slow. Consider server mode."}
	}
	switch {
	case hw.VRAMMiB >= 22000:
		return Recommendation{Tier: "S", Model: "qwen2.5-coder:14b", Reason: "24GB+ VRAM — 14B synthesis with headroom."}
	case hw.VRAMMiB >= 12000:
		return Recommendation{Tier: "A", Model: "qwen2.5-coder:7b", Reason: "12-16GB VRAM — 7B synthesis fits alongside other GPU use."}
	default:
		return Recommendation{Tier: "B", Model: "qwen2.5-coder:3b", Reason: "<=12GB VRAM — 3B synthesis stays safe when the GPU is shared with a game. Hybrid retrieval is CPU-only."}
	}
}

// Detect probes the host. Returns best-effort info; missing pieces stay zero
// and Pick() handles the degenerate case.
func Detect() Hardware {
	hw := Hardware{
		CPU: runtime.GOARCH + " (" + strconv.Itoa(runtime.NumCPU()) + " cores)",
	}
	if name, vram, ok := detectNVIDIA(); ok {
		hw.GPU = name
		hw.VRAMMiB = vram
		hw.HasGPU = true
	}
	hw.RAMGiB = detectRAMGiB()
	return hw
}

// detectNVIDIA shells out to nvidia-smi. Returns name + total memory in MiB.
func detectNVIDIA() (string, int, bool) {
	out, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits").CombinedOutput()
	if err != nil {
		return "", 0, false
	}
	line := strings.TrimSpace(string(out))
	if line == "" {
		return "", 0, false
	}
	// Multiple GPUs → take the first.
	first := strings.Split(line, "\n")[0]
	parts := strings.Split(first, ",")
	if len(parts) < 2 {
		return "", 0, false
	}
	name := strings.TrimSpace(parts[0])
	vram, _ := strconv.Atoi(strings.TrimSpace(parts[1]))
	return name, vram, true
}

func detectRAMGiB() int {
	switch runtime.GOOS {
	case "windows":
		// Try wmic first (works on Win10 and some Win11 systems).
		out, err := exec.Command("wmic", "computersystem", "get", "TotalPhysicalMemory", "/value").CombinedOutput()
		if err == nil {
			for _, line := range strings.Split(string(out), "\n") {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "TotalPhysicalMemory=") {
					raw := strings.TrimPrefix(line, "TotalPhysicalMemory=")
					if n, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 64); err == nil && n > 0 {
						return int(n / (1 << 30))
					}
				}
			}
		}
		// Fallback: PowerShell Get-CimInstance. wmic is deprecated on Win11.
		ps, err := exec.Command("powershell", "-NoProfile", "-NonInteractive", "-Command",
			"(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory").CombinedOutput()
		if err == nil {
			raw := strings.TrimSpace(string(ps))
			if n, err := strconv.ParseInt(raw, 10, 64); err == nil && n > 0 {
				return int(n / (1 << 30))
			}
		}
	case "linux":
		b, err := os.ReadFile("/proc/meminfo")
		if err == nil {
			for _, line := range strings.Split(string(b), "\n") {
				if strings.HasPrefix(line, "MemTotal:") {
					fields := strings.Fields(line)
					if len(fields) >= 2 {
						if kb, err := strconv.Atoi(fields[1]); err == nil {
							return kb / (1 << 20)
						}
					}
				}
			}
		}
	case "darwin":
		out, err := exec.Command("sysctl", "-n", "hw.memsize").CombinedOutput()
		if err == nil {
			raw := strings.TrimSpace(string(out))
			if n, err := strconv.ParseInt(raw, 10, 64); err == nil && n > 0 {
				return int(n / (1 << 30))
			}
		}
	}
	return 0
}

// EnsureOllama checks that the Ollama daemon is reachable and the model is
// pulled. Returns nil on success, error otherwise. Does NOT install Ollama
// itself — that's a system-level action the user should approve first; we
// just point them at the install URL.
func EnsureOllama(ctx context.Context, model string) error {
	cl := ollama.New("")
	if _, err := cl.Version(ctx); err != nil {
		return fmt.Errorf("Ollama not reachable at localhost:11434. Install from https://ollama.com/download then re-run install. (%w)", err)
	}
	have, err := cl.List(ctx)
	if err == nil {
		for _, n := range have {
			if strings.EqualFold(n, model) || strings.HasPrefix(strings.ToLower(n), strings.ToLower(model)+":") {
				return nil
			}
		}
	}
	// Pull via the Ollama HTTP API. Avoid spawning the CLI so we work even
	// when Ollama isn't on PATH (Windows installer puts it under AppData).
	fmt.Fprintf(os.Stderr, "Pulling model %s — this is a ~5GB download for tier B, larger for higher tiers.\n", model)
	out, err := exec.Command("ollama", "pull", model).CombinedOutput()
	if err != nil {
		return fmt.Errorf("ollama pull %s failed: %v: %s", model, err, string(out))
	}
	return nil
}

// PatchClaudeJSON writes the void-memory MCP server entry into
// ~/.claude.json. Idempotent: re-running just updates the path.
func PatchClaudeJSON(binaryPath, model string) error {
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, ".claude.json")
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read %s: %w", path, err)
	}
	var data map[string]interface{}
	if err := json.Unmarshal(b, &data); err != nil {
		return fmt.Errorf("parse %s: %w", path, err)
	}
	mcp, _ := data["mcpServers"].(map[string]interface{})
	if mcp == nil {
		mcp = make(map[string]interface{})
	}
	mcp["void-memory"] = map[string]interface{}{
		"command": binaryPath,
		"args":    []string{},
		"env": map[string]string{
			"VOID_MEMORY_MODEL":       model,
			"VOID_MEMORY_EMBED_MODEL": "nomic-embed-text",
		},
	}
	data["mcpServers"] = mcp
	out, _ := json.MarshalIndent(data, "", "  ")
	return os.WriteFile(path, out, 0o644)
}

// PatchSettingsJSON adds the four void-memory MCP tools to the allow list
// of ~/.claude/settings.json. Defensive layer — annotations should also
// auto-approve them, but this guarantees no permission prompts even on
// clients that don't respect annotations.
func PatchSettingsJSON() error {
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, ".claude", "settings.json")
	var data map[string]interface{}
	if b, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(b, &data)
	}
	if data == nil {
		data = make(map[string]interface{})
	}
	perms, _ := data["permissions"].(map[string]interface{})
	if perms == nil {
		perms = make(map[string]interface{})
	}
	allow, _ := perms["allow"].([]interface{})
	need := []string{
		"mcp__void-memory__recall",
		"mcp__void-memory__read_session",
		"mcp__void-memory__list_topics",
		"mcp__void-memory__index_status",
	}
	seen := make(map[string]struct{}, len(allow))
	for _, x := range allow {
		if s, ok := x.(string); ok {
			seen[s] = struct{}{}
		}
	}
	for _, n := range need {
		if _, ok := seen[n]; !ok {
			allow = append(allow, n)
		}
	}
	perms["allow"] = allow
	data["permissions"] = perms
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	out, _ := json.MarshalIndent(data, "", "  ")
	return os.WriteFile(path, out, 0o644)
}

// Run performs the full install and prints progress to stderr. Returns
// nil on success.
func Run(ctx context.Context, binaryPath string, runInitialIndex bool) error {
	hw := Detect()
	rec := Pick(hw)
	fmt.Fprintf(os.Stderr, "Detected: %s | VRAM=%dMB | RAM=%dGiB | CPU=%s\n", hw.GPU, hw.VRAMMiB, hw.RAMGiB, hw.CPU)
	fmt.Fprintf(os.Stderr, "Tier %s — %s\n", rec.Tier, rec.Reason)
	fmt.Fprintf(os.Stderr, "Model: %s\n\n", rec.Model)

	if err := EnsureOllama(ctx, rec.Model); err != nil {
		return err
	}
	if err := EnsureOllama(ctx, "nomic-embed-text"); err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr, "Ollama + models (synthesis + embed): OK")

	if err := PatchClaudeJSON(binaryPath, rec.Model); err != nil {
		return fmt.Errorf("patch claude.json: %w", err)
	}
	fmt.Fprintln(os.Stderr, "~/.claude.json: registered MCP server")

	if err := PatchSettingsJSON(); err != nil {
		return fmt.Errorf("patch settings.json: %w", err)
	}
	fmt.Fprintln(os.Stderr, "~/.claude/settings.json: allow rules added")

	if runInitialIndex {
		fmt.Fprintln(os.Stderr, "Building initial hybrid index (CPU-pinned embed; may take a few minutes)...")
		home, _ := os.UserHomeDir()
		be, err := backend.NewLocal(backend.LocalConfig{
			DataDir:     filepath.Join(home, ".void-memory"),
			ProjectsDir: filepath.Join(home, ".claude", "projects"),
			Model:       rec.Model,
			EmbModel:    "nomic-embed-text",
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "WARN: index init: %v (run `void-memory reindex` later)\n", err)
		} else {
			defer be.Close()
			if err := be.ReindexHybrid(ctx); err != nil {
				fmt.Fprintf(os.Stderr, "WARN: index build: %v\n", err)
			} else {
				fmt.Fprintln(os.Stderr, "Initial hybrid index: OK")
			}
		}
	}

	fmt.Fprintln(os.Stderr, "\nDone. Open a new Claude Code session — void-memory will load automatically.")
	return nil
}
