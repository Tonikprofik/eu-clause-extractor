from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

# Ensure project root is on sys.path so `src` is importable when running `uvicorn api.main:app`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Pipelines and schema
try:
    from src.pipelines.rag_pipeline import (
        run_rag_clause_extraction,
        CLAUSE_TYPES as BASE_CLAUSE_TYPES,
    )
except Exception:
    run_rag_clause_extraction = None  # type: ignore
    BASE_CLAUSE_TYPES = [
        "Subject Matter & Scope",
        "Definitions",
        "Obligations of Member States",
        "Penalties",
        "Entry into Force & Application",
    ]

try:
    from src.pipelines.advanced_rag_pipeline import run_advanced_rag_pipeline
except Exception:
    run_advanced_rag_pipeline = None  # type: ignore

# Configuration
API_TITLE = "EU Clause Extractor API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
Extract legal clauses from EU regulations using RAG-powered LLM pipelines.

## Features

- **Semantic Retrieval**: ChromaDB vector search for relevant document chunks
- **Model Flexibility**: LiteLLM proxy for unified access to Claude, GPT-4, Gemma, etc.
- **Observability**: Langfuse integration for tracing and evaluation
- **HyDE Mode**: Optional Hypothetical Document Embedding for improved retrieval

## Clause Types

The extractor identifies these clause categories:
- Subject Matter & Scope
- Definitions  
- Obligations of Member States
- Penalties
- Entry into Force & Application

## Links

- [GitHub Repository](https://github.com/Tonikprofik/eu-clause-extractor)
- [Langfuse](https://langfuse.com) for observability
"""

LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
DEFAULT_READER_MODEL = os.getenv("RAG_READER_MODEL_ALIAS", "claude-3-7")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_ALIAS", "text-embedding-3-small")
DEFAULT_USE_HYDE = os.getenv("DEFAULT_USE_HYDE", "false").lower() == "true"
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "150"))

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_ENABLED = bool(LANGFUSE_HOST and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://app.eucontracts.ai,http://localhost:3000").split(",")

# Prometheus metrics
REQ_HISTOGRAM = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["route", "method", "status"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5, 10, float("inf")),
)
RAG_STAGE_HISTOGRAM = Histogram(
    "rag_stage_duration_seconds",
    "RAG stage duration in seconds",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, float("inf")),
)
RAG_ERRORS = Counter(
    "rag_errors_total",
    "Total RAG errors by stage and exception class",
    ["stage", "exception_class"],
)
LLM_TOKENS = Counter(
    "llm_tokens_total",
    "LLM tokens observed, by model and role",
    ["model", "role"],
)

# OpenAPI Tags
tags_metadata = [
    {"name": "extraction", "description": "Core clause extraction endpoints"},
    {"name": "meta", "description": "Health checks and model discovery"},
    {"name": "observability", "description": "Prometheus metrics for monitoring"},
]

# FastAPI app with enhanced metadata
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    openapi_tags=tags_metadata,
    contact={
        "name": "Tony Thai Do",
        "url": "https://github.com/Tonikprofik/eu-clause-extractor",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_http_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration = time.perf_counter() - start
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        method = request.method
        status = str(getattr(response, "status_code", 500))
        REQ_HISTOGRAM.labels(route_path, method, status).observe(duration)


# ============================================================================
# Pydantic Models with Examples
# ============================================================================

class ExtractionOptions(BaseModel):
    """Configuration options for the extraction pipeline."""
    
    use_hyde: bool = Field(
        default=False,
        description="Enable HyDE (Hypothetical Document Embedding) for improved retrieval quality",
        json_schema_extra={"example": False},
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to retrieve",
        json_schema_extra={"example": 5},
    )
    reader_model: Optional[str] = Field(
        default=None,
        description="LLM model alias for clause extraction (e.g., claude-3-7, gpt-4)",
        json_schema_extra={"example": "claude-3-7"},
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model alias for semantic search",
        json_schema_extra={"example": "text-embedding-3-small"},
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "use_hyde": False,
                "top_k": 5,
                "reader_model": "claude-3-7",
                "embedding_model": "text-embedding-3-small",
            }
        }
    }


class ExtractionRequest(BaseModel):
    """Request body for clause extraction."""
    
    document_id: str = Field(
        description="Unique identifier for the document (e.g., CELEX number)",
        json_schema_extra={"example": "32016R0679"},
    )
    document_text: str = Field(
        description="Full text of the EU regulation to analyze",
        json_schema_extra={"example": "Article 1\nSubject matter and scope\nThis Regulation lays down rules relating to the protection of natural persons..."},
    )
    user_query: str = Field(
        description="Natural language query describing what to extract",
        json_schema_extra={"example": "Extract definitions, obligations, and penalties from this regulation"},
    )
    clause_types: Optional[List[str]] = Field(
        default=None,
        description="Specific clause types to extract (defaults to all types)",
        json_schema_extra={"example": ["Definitions", "Penalties"]},
    )
    language: str = Field(
        default="en",
        description="Document language (en, de)",
        json_schema_extra={"example": "en"},
    )
    options: ExtractionOptions = Field(
        default_factory=ExtractionOptions,
        description="Pipeline configuration options",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": "32016R0679",
                "document_text": "Article 1\nSubject matter and scope\nThis Regulation lays down rules relating to the protection of natural persons with regard to the processing of personal data.\n\nArticle 4\nDefinitions\n(1) 'personal data' means any information relating to an identified or identifiable natural person ('data subject');\n(2) 'processing' means any operation performed on personal data.\n\nArticle 83\nPenalties\nMember States shall lay down the rules on penalties applicable to infringements of this Regulation.",
                "user_query": "Extract definitions and penalties from this GDPR excerpt",
                "clause_types": ["Definitions", "Penalties"],
                "language": "en",
                "options": {"use_hyde": False, "top_k": 5, "reader_model": "claude-3-7"},
            }
        }
    }


class PredictedAnnotation(BaseModel):
    """A single extracted clause."""
    
    clause_type: str = Field(
        description="Category of the extracted clause",
        json_schema_extra={"example": "Definitions"},
    )
    clause_text: str = Field(
        description="The extracted clause text",
        json_schema_extra={"example": "'personal data' means any information relating to an identified or identifiable natural person"},
    )


class ExtractionResponse(BaseModel):
    """Response from clause extraction."""
    
    predicted_annotations: List[PredictedAnnotation] = Field(
        description="List of extracted clauses with their types",
    )
    retrieved_chunks: List[str] = Field(
        description="Document chunks used for context during extraction",
    )
    reader_llm_output_raw: Optional[str] = Field(
        default=None,
        description="Raw LLM output before parsing (for debugging)",
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Langfuse trace ID for observability",
        json_schema_extra={"example": "abc123-def456"},
    )
    timings: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Timing breakdown by pipeline stage (ms)",
    )
    usage: Dict[str, Optional[int]] = Field(
        default_factory=dict,
        description="Token usage statistics",
    )
    model_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Models used for this extraction",
    )
    error: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error details if extraction failed",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_annotations": [
                    {"clause_type": "Definitions", "clause_text": "'personal data' means any information relating to an identified or identifiable natural person"},
                    {"clause_type": "Penalties", "clause_text": "Member States shall lay down the rules on penalties applicable to infringements"},
                ],
                "retrieved_chunks": ["Article 4\nDefinitions\n(1) 'personal data' means...", "Article 83\nPenalties\nMember States shall..."],
                "trace_id": "abc123-def456",
                "timings": {"read_ms": 2340.5},
                "usage": {"input_tokens": 1250, "output_tokens": 180},
                "model_info": {"reader_model": "claude-3-7", "embedding_model": "text-embedding-3-small", "use_hyde": False},
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(description="Overall status: ok or degraded")
    litellm: Dict[str, Any] = Field(description="LiteLLM proxy status")
    langfuse: Dict[str, Any] = Field(description="Langfuse observability status")
    vector: Dict[str, Any] = Field(description="Vector store status")
    defaults: Dict[str, Any] = Field(description="Default configuration values")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "litellm": {"url": "http://localhost:4000", "ok": True, "status_code": 200},
                "langfuse": {"host": "https://cloud.langfuse.com", "enabled": True},
                "vector": {"backend": "chroma", "mode": "in-process"},
                "defaults": {"reader_model": "claude-3-7", "embedding_model": "text-embedding-3-small", "use_hyde": False},
            }
        }
    }


class ModelsResponse(BaseModel):
    """Available models response."""
    
    reader_models: List[str] = Field(
        description="LLM models available for clause extraction",
        json_schema_extra={"example": ["claude-3-7", "gpt-4", "gemma3-4b"]},
    )
    embedding_models: List[str] = Field(
        description="Embedding models available for semantic search",
        json_schema_extra={"example": ["text-embedding-3-small", "text-embedding-3-large"]},
    )
    all_models: List[str] = Field(description="All configured model aliases")
    defaults: Dict[str, str] = Field(
        description="Default model selections",
        json_schema_extra={"example": {"reader_model": "claude-3-7", "embedding_model": "text-embedding-3-small"}},
    )


# ============================================================================
# Helper Functions
# ============================================================================

def load_litellm_models(
    config_path: str = os.path.join("src", "config", "litellm_config.yaml"),
) -> Tuple[List[str], List[str], List[str]]:
    """Load model aliases from LiteLLM config and categorize reader vs embedding."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        model_list = data.get("model_list", [])
        all_models = [m.get("model_name") for m in model_list if isinstance(m, dict) and m.get("model_name")]
        reader_models: List[str] = []
        embedding_models: List[str] = []
        for m in model_list:
            if not isinstance(m, dict):
                continue
            name = m.get("model_name", "")
            params = m.get("litellm_params", {}) or {}
            underlying = str(params.get("model", "")).lower()
            if "embedding" in name.lower() or "embedding" in underlying:
                embedding_models.append(name)
            else:
                reader_models.append(name)
        return list(dict.fromkeys(reader_models)), list(dict.fromkeys(embedding_models)), list(dict.fromkeys(all_models))
    except Exception:
        return ([DEFAULT_READER_MODEL], [DEFAULT_EMBEDDING_MODEL], [DEFAULT_READER_MODEL, DEFAULT_EMBEDDING_MODEL])


def ensure_pipelines_available():
    if run_rag_clause_extraction is None or run_advanced_rag_pipeline is None:
        raise RuntimeError("Pipeline modules not importable. Run from repository root.")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/api/v1/health", response_model=HealthResponse, tags=["meta"], summary="Health check",
         description="Check API health and upstream service status (LiteLLM, Langfuse, vector store).")
def health() -> Dict[str, Any]:
    lite_status = {"url": LITELLM_PROXY_URL, "ok": False}
    try:
        r = requests.get(f"{LITELLM_PROXY_URL.rstrip('/')}/v1/models", timeout=5)
        lite_status["ok"] = r.status_code < 500
        lite_status["status_code"] = r.status_code
    except Exception:
        lite_status["error"] = str(e)
    return {
        "status": "ok" if lite_status.get("ok") else "degraded",
        "litellm": lite_status,
        "langfuse": {"host": LANGFUSE_HOST, "enabled": LANGFUSE_ENABLED},
        "vector": {"backend": "chroma", "mode": "in-process"},
        "defaults": {"reader_model": DEFAULT_READER_MODEL, "embedding_model": DEFAULT_EMBEDDING_MODEL, "use_hyde": DEFAULT_USE_HYDE},
    }


@app.get("/api/v1/models", response_model=ModelsResponse, tags=["meta"], summary="List available models",
         description="Retrieve all configured LLM and embedding models from LiteLLM proxy.")
def list_models():
    readers, embeddings, all_models = load_litellm_models()
    return {
        "reader_models": readers,
        "embedding_models": embeddings,
        "all_models": all_models,
        "defaults": {"reader_model": DEFAULT_READER_MODEL, "embedding_model": DEFAULT_EMBEDDING_MODEL},
    }


@app.get("/metrics", tags=["observability"], summary="Prometheus metrics",
         description="Prometheus-format metrics for monitoring.", response_class=Response,
         responses={200: {"content": {"text/plain": {}}, "description": "Prometheus metrics"}})
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/extract-clauses", response_model=ExtractionResponse, tags=["extraction"],
          summary="Extract clauses from EU regulation",
          description="Extract legal clauses from EU regulation text using RAG-powered LLM pipelines.\n\n**Pipeline Options:**\n- Standard RAG (default)\n- HyDE Mode (use_hyde=true) for improved retrieval",
          responses={200: {"description": "Successful extraction"}, 422: {"description": "Validation error"}, 500: {"description": "Pipeline error"}})
def extract_clauses(req: ExtractionRequest) -> ExtractionResponse:
    ensure_pipelines_available()
    opts = req.options or ExtractionOptions()
    clause_types = req.clause_types or BASE_CLAUSE_TYPES
    timings: Dict[str, Optional[float]] = {"chunk_ms": None, "embed_ms": None, "retrieve_ms": None, "read_ms": None}
    usage: Dict[str, Optional[int]] = {"input_tokens": None, "output_tokens": None}
    model_info = {
        "reader_model": opts.reader_model or DEFAULT_READER_MODEL,
        "embedding_model": opts.embedding_model or DEFAULT_EMBEDDING_MODEL,
        "use_hyde": bool(opts.use_hyde),
    }
    try:
        t0 = time.perf_counter()
        if opts.use_hyde:
            result = run_advanced_rag_pipeline(
                document_text=req.document_text, document_id=req.document_id, user_query=req.user_query,
                target_clause_types=clause_types, language=req.language,
                reader_model_override=model_info["reader_model"], embedding_model_override=model_info["embedding_model"],
            )
            predicted = result.get("predicted_annotations", []) or []
            retrieved = result.get("retrieved_chunks", []) or []
            raw = result.get("reader_llm_output_raw")
            trace_id = result.get("langfuse_trace_id")
        else:
            result = run_rag_clause_extraction(
                document_text=req.document_text, document_id=req.document_id, language=req.language,
                reader_model_alias_param=model_info["reader_model"], embedding_model_alias_param=model_info["embedding_model"],
                litellm_proxy_url_param=LITELLM_PROXY_URL,
            )
            predicted = result.get("predicted_annotations", []) or []
            retrieved = result.get("retrieved_chunk_texts", []) or []
            raw = None
            trace_id = result.get("rag_trace_id")
        t1 = time.perf_counter()
        timings["read_ms"] = (t1 - t0) * 1000.0
        RAG_STAGE_HISTOGRAM.labels("read").observe(t1 - t0)
        predicted_items = [PredictedAnnotation(**p) for p in predicted if isinstance(p, dict) and "clause_type" in p and "clause_text" in p]
        return ExtractionResponse(
            predicted_annotations=predicted_items, retrieved_chunks=retrieved, reader_llm_output_raw=raw,
            trace_id=trace_id, timings=timings, usage=usage, model_info=model_info,
        )
    except HTTPException:
        raise
    except Exception:
        RAG_ERRORS.labels("pipeline", e.__class__.__name__).inc()
        raise HTTPException(status_code=500, detail={"message": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
