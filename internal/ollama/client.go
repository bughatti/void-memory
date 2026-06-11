// Package ollama is a small HTTP client for the local Ollama daemon
// (https://ollama.com/). We talk to it on localhost:11434 by default. The
// client is intentionally minimal — only what void-memory needs.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const defaultBaseURL = "http://localhost:11434"

// Client is a minimal Ollama HTTP client.
type Client struct {
	baseURL string
	http    *http.Client
}

// New returns a Client pointing at the configured Ollama URL. baseURL of ""
// means use the default localhost:11434.
func New(baseURL string) *Client {
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Client{
		baseURL: baseURL,
		http: &http.Client{
			// Generation can take a while on tier C hardware; pad generously.
			Timeout: 120 * time.Second,
		},
	}
}

// Version checks /api/version. Used at startup to detect Ollama is running.
type VersionResponse struct {
	Version string `json:"version"`
}

func (c *Client) Version(ctx context.Context) (string, error) {
	req, _ := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/api/version", nil)
	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama unreachable: %w", err)
	}
	defer resp.Body.Close()
	var v VersionResponse
	if err := json.NewDecoder(resp.Body).Decode(&v); err != nil {
		return "", err
	}
	return v.Version, nil
}

// List returns names of locally-installed models. Used to verify the
// configured model is pulled.
type listResp struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

func (c *Client) List(ctx context.Context) ([]string, error) {
	req, _ := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/api/tags", nil)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var lr listResp
	if err := json.NewDecoder(resp.Body).Decode(&lr); err != nil {
		return nil, err
	}
	out := make([]string, len(lr.Models))
	for i, m := range lr.Models {
		out[i] = m.Name
	}
	return out, nil
}

// GenerateRequest mirrors Ollama's /api/generate body. We don't expose every
// option — just the fields void-memory needs.
type GenerateRequest struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	System  string                 `json:"system,omitempty"`
	Stream  bool                   `json:"stream"`
	Format  string                 `json:"format,omitempty"` // "json" forces structured output
	Options map[string]interface{} `json:"options,omitempty"`
}

// GenerateResponse mirrors the non-streaming response.
type GenerateResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
	// Skipping the timing/eval fields — we don't use them yet.
}

// Generate runs a single-shot completion. Stream=false; for void-memory we
// want the full response in one shot so we can hand it back to MCP.
func (c *Client) Generate(ctx context.Context, req GenerateRequest) (string, error) {
	req.Stream = false
	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	httpReq, _ := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/generate", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("ollama generate: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama generate %d: %s", resp.StatusCode, string(b))
	}
	var gr GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&gr); err != nil {
		return "", err
	}
	return gr.Response, nil
}

// GenerateJSON is a convenience for structured output. Pass a struct
// pointer; the LLM is constrained to JSON and we Unmarshal into it.
//
// num_predict is set high so structured outputs of reasonable size aren't
// truncated mid-write; Ollama's default of 128 tokens is far too small for
// our extraction/routing prompts.
func (c *Client) GenerateJSON(ctx context.Context, model, system, prompt string, out interface{}) error {
	resp, err := c.Generate(ctx, GenerateRequest{
		Model:  model,
		Prompt: prompt,
		System: system,
		Format: "json",
		Options: map[string]interface{}{
			"temperature": 0.1,
			"num_predict": 2048,
			"num_ctx":     16384,
		},
	})
	if err != nil {
		return err
	}
	return json.Unmarshal([]byte(resp), out)
}
