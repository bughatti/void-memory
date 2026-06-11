// uninstall.go reverses what install.go did: removes the MCP server entry
// from ~/.claude.json, drops the allow rules from ~/.claude/settings.json,
// and optionally wipes ~/.void-memory data (transcripts unchanged — those
// belong to Claude Code, not us).
//
// We do NOT uninstall Ollama or the model — those are useful for other
// applications and we shouldn't presume to remove them.
package install

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Uninstall undoes the install steps. If wipeDataDir is true, also removes
// ~/.void-memory entirely (catalog, session metadata, config).
func Uninstall(wipeDataDir bool) error {
	home, _ := os.UserHomeDir()

	if err := unpatchClaudeJSON(home); err != nil {
		fmt.Fprintf(os.Stderr, "WARN: ~/.claude.json: %v\n", err)
	} else {
		fmt.Fprintln(os.Stderr, "~/.claude.json: removed MCP server entry")
	}

	if err := unpatchSettingsJSON(home); err != nil {
		fmt.Fprintf(os.Stderr, "WARN: ~/.claude/settings.json: %v\n", err)
	} else {
		fmt.Fprintln(os.Stderr, "~/.claude/settings.json: removed allow rules")
	}

	if wipeDataDir {
		dir := filepath.Join(home, ".void-memory")
		if err := os.RemoveAll(dir); err != nil {
			fmt.Fprintf(os.Stderr, "WARN: remove %s: %v\n", dir, err)
		} else {
			fmt.Fprintf(os.Stderr, "%s: removed\n", dir)
		}
	} else {
		fmt.Fprintf(os.Stderr, "~/.void-memory data preserved (pass --wipe to remove)\n")
	}

	fmt.Fprintln(os.Stderr, "\nDone. Ollama and the local model are unchanged — remove them with `ollama rm` if desired.")
	return nil
}

func unpatchClaudeJSON(home string) error {
	path := filepath.Join(home, ".claude.json")
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var data map[string]interface{}
	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}
	if mcp, ok := data["mcpServers"].(map[string]interface{}); ok {
		delete(mcp, "void-memory")
		if len(mcp) == 0 {
			delete(data, "mcpServers")
		} else {
			data["mcpServers"] = mcp
		}
	}
	out, _ := json.MarshalIndent(data, "", "  ")
	return os.WriteFile(path, out, 0o644)
}

func unpatchSettingsJSON(home string) error {
	path := filepath.Join(home, ".claude", "settings.json")
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var data map[string]interface{}
	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}
	perms, _ := data["permissions"].(map[string]interface{})
	if perms == nil {
		return nil
	}
	allow, _ := perms["allow"].([]interface{})
	if allow == nil {
		return nil
	}
	drop := map[string]struct{}{
		"mcp__void-memory__recall":       {},
		"mcp__void-memory__read_session": {},
		"mcp__void-memory__list_topics":  {},
		"mcp__void-memory__index_status": {},
	}
	filtered := allow[:0]
	for _, v := range allow {
		if s, ok := v.(string); ok {
			if _, gone := drop[s]; gone {
				continue
			}
		}
		filtered = append(filtered, v)
	}
	perms["allow"] = filtered
	data["permissions"] = perms
	out, _ := json.MarshalIndent(data, "", "  ")
	return os.WriteFile(path, out, 0o644)
}
