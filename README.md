# EU Clause Extractor

**MSc Thesis Project** — Legal clause extraction from EU regulations using LLM pipelines with progressive complexity.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenAPI](https://img.shields.io/badge/OpenAPI-3.1-6BA539.svg)](api/openapi.yaml)

## Overview

A multi-stage LLM pipeline for extracting legal clauses from EU regulations, demonstrating modern LLMOps practices:

- **Observability**: End-to-end tracing with [Langfuse](https://langfuse.com)
- **Model Routing**: Unified API via [LiteLLM](https://github.com/BerriAI/litellm) proxy
- **Vector Search**: ChromaDB for semantic retrieval
- **Agentic Patterns**: Self-critique and refinement with LangGraph

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         EU Regulation                           │
│                     (EUR-Lex Document)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────┐        ┌──────────────┐
   │ Baseline │        │   RAG    │        │   Agentic    │
   │ (Direct) │        │ Pipeline │        │   Pipeline   │
   └──────────┘        └──────────┘        └──────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   LLM extracts         ChromaDB            RAG extraction
   clauses directly     retrieves           + LLM critique
                        relevant chunks     + refinement loop
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                    ┌───────────────────┐
                    │   Langfuse        │
                    │   (Tracing &      │
                    │    Evaluation)    │
                    └───────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for LiteLLM proxy and Langfuse)
- API keys: OpenAI, Anthropic (optional: local Ollama)

### Setup

```bash
# Clone and setup
git clone https://github.com/Tonikprofik/eu-clause-extractor.git
cd eu-clause-extractor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Environment variables
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY
# - LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

# Start LiteLLM proxy
docker run -v "${PWD}/src/config/litellm_config.yaml:/app/config.yaml" \
  -p 4000:4000 ghcr.io/berriai/litellm:main-latest \
  --config /app/config.yaml

# Start Langfuse (optional, for tracing)
cd langfuse && docker compose up -d
```

### Run Extraction

```python
from src.pipelines.rag_pipeline import run_rag_clause_extraction

result = run_rag_clause_extraction(
    document_text="Your EU regulation text here...",
    document_id="32012R0685",
    reader_model_alias="claude-3-7",
    embedder_model_alias="text-embedding-3-small",
)
print(result["clauses"])
```

## API Reference

The extraction service exposes a REST API on port `8080`. Full specification: [api/openapi.yaml](api/openapi.yaml)

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/extract-clauses` | Extract legal clauses from EU regulation text |
| `GET` | `/api/v1/health` | Health check with upstream service status |
| `GET` | `/api/v1/models` | List available LLM and embedding models |
| `GET` | `/metrics` | Prometheus metrics for monitoring |

### Example Request

```bash
curl -X POST http://localhost:8080/api/v1/extract-clauses \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "32016R0679",
    "document_text": "Article 4\nDefinitions\n(1) '\''personal data'\'' means any information...",
    "user_query": "Extract definitions from this GDPR excerpt",
    "clause_types": ["Definitions"],
    "options": {
      "top_k": 5,
      "reader_model": "claude-3-7"
    }
  }'
```

### Example Response

```json
{
  "predicted_annotations": [
    {
      "clause_type": "Definitions",
      "clause_text": "'personal data' means any information relating to an identified natural person"
    }
  ],
  "retrieved_chunks": ["Article 4\nDefinitions\n(1) 'personal data' means..."],
  "trace_id": "abc123-def456",
  "timings": { "read_ms": 2340.5 },
  "usage": { "input_tokens": 1250, "output_tokens": 180 },
  "model_info": {
    "reader_model": "claude-3-7",
    "embedding_model": "text-embedding-3-small"
  }
}
```

### Interactive Docs

When running the API locally, interactive documentation is available at:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

## Project Structure

```
src/
├── pipelines/              # Core extraction engines
│   ├── rag_pipeline.py     # RAG-based extraction
│   ├── agentic_extractor.py    # LangGraph agent with critique
│   └── advanced_rag_pipeline.py # HyDE + advanced retrieval
├── evaluation/             # Metrics and evaluation runners
├── data/                   # Data processing utilities
└── config/
    └── litellm_config.yaml # Model routing configuration

api/                        # FastAPI service
├── main.py                 # API implementation
└── openapi.yaml            # OpenAPI 3.1 specification
examples/                   # Jupyter notebooks and cookbooks
gold_annotations/           # Ground truth dataset
ui/                         # Next.js 15 web interface
```

## Evaluation Results

| Pipeline | F1 | Precision | Recall | Notes |
|----------|---:|----------:|-------:|-------|
| **Baseline** (Claude 3.5) | 0.893 | 0.895 | 0.895 | Direct extraction, no retrieval |
| **RAG** (Gemma-3, K=3) | 0.735 | 0.611 | 0.963 | High recall, lower precision |
| **RAG** (GPT-4.1-mini, K=5) | 0.686 | 0.654 | 0.750 | Balanced performance |
| **Agent** (Gemma3+Critique) | 0.430 | 0.509 | 0.432 | Self-critique loop |

Full results: [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md)

## Key Findings

1. **Baseline outperforms RAG** on this task — the complete document context helps LLMs identify clause boundaries better than chunked retrieval
2. **RAG excels at recall** — retrieval successfully surfaces relevant passages, but chunking loses structural context
3. **Agentic critique adds overhead** — the self-critique loop improves quality on complex documents but reduces overall F1 on simpler regulations
4. **Model choice matters less than architecture** — GPT-4.1-mini and Claude 3.7 perform similarly within each pipeline architecture

## Technologies

| Component | Technology |
|-----------|------------|
| LLM Gateway | LiteLLM |
| Observability | Langfuse |
| Vector Store | ChromaDB |
| Agent Framework | LangGraph |
| API | FastAPI |
| UI | Next.js 15 + shadcn/ui |

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*MSc Thesis Project — Aalborg University, 2025*
