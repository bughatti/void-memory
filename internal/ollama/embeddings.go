package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// embeddingsRequest mirrors Ollama's /api/embeddings body (single-prompt form,
// the widely-supported endpoint).
type embeddingsRequest struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	Options map[string]interface{} `json:"options,omitempty"`
}

type embeddingsResponse struct {
	Embedding []float64 `json:"embedding"`
}

// Embeddings returns the embedding vector for text from the given model. Used
// by the hybrid-retrieval core to vectorize chunks (at index time) and queries
// (at search time). Run a small CPU embedding model here (e.g. nomic-embed-text
// or all-minilm) — it must be a model pulled for embeddings, NOT the synthesis
// LLM. Returned as float32; the retrieval layer quantizes to binary and keeps
// the float copy for rescoring.
func (c *Client) Embeddings(ctx context.Context, model, text string) ([]float32, error) {
	return c.embeddings(ctx, model, text, nil)
}

// EmbeddingsCPU forces CPU inference (num_gpu=0). Use for batch reindex so the
// embedding job never competes with a foreground GPU app (e.g. a game) for
// VRAM — on an 8 GB card that contention can crash the display driver. Latency
// is irrelevant for a background index build, so CPU is the safe default there.
func (c *Client) EmbeddingsCPU(ctx context.Context, model, text string) ([]float32, error) {
	return c.embeddings(ctx, model, text, map[string]interface{}{"num_gpu": 0})
}

func (c *Client) embeddings(ctx context.Context, model, text string, opts map[string]interface{}) ([]float32, error) {
	body, err := json.Marshal(embeddingsRequest{Model: model, Prompt: text, Options: opts})
	if err != nil {
		return nil, err
	}
	req, _ := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/embeddings", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama embeddings: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama embeddings %d: %s", resp.StatusCode, string(b))
	}
	var er embeddingsResponse
	if err := json.NewDecoder(resp.Body).Decode(&er); err != nil {
		return nil, err
	}
	if len(er.Embedding) == 0 {
		return nil, fmt.Errorf("ollama embeddings: empty vector for model %q", model)
	}
	out := make([]float32, len(er.Embedding))
	for i, v := range er.Embedding {
		out[i] = float32(v)
	}
	return out, nil
}
