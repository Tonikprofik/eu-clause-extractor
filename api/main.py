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
except Exception as e:
    # Fallback to allow app import; actual endpoint will raise on call
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
API_TITLE = "EUContracts API"
API_VERSION = "v1"
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

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://app.eucontracts.ai").split(",")

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
    ["stage"],  # chunk, embed, retrieve, read
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

# FastAPI app
app = FastAPI(title=API_TITLE, version=API_VERSION)
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
    # Resolve route template if available after response
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


# Models
class ExtractionOptions(BaseModel):
    use_hyde: bool = Field(
        default=DEFAULT_USE_HYDE, description="Enable HyDE (quality mode)"
    )
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)
    reader_model: Optional[str] = Field(default=DEFAULT_READER_MODEL)
    embedding_model: Optional[str] = Field(default=DEFAULT_EMBEDDING_MODEL)


class ExtractionRequest(BaseModel):
    document_id: str
    document_text: str
    user_query: str
    clause_types: Optional[List[str]] = None
    language: str = "en"
    options: ExtractionOptions = Field(default_factory=ExtractionOptions)


class PredictedAnnotation(BaseModel):
    clause_type: str
    clause_text: str


class ExtractionResponse(BaseModel):
    predicted_annotations: List[PredictedAnnotation]
    retrieved_chunks: List[str]
    reader_llm_output_raw: Optional[str] = None
    trace_id: Optional[str] = None
    timings: Dict[str, Optional[float]] = {}
    usage: Dict[str, Optional[int]] = {}
    model_info: Dict[str, Any] = {}
    error: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    reader_models: List[str]
    embedding_models: List[str]
    all_models: List[str]
    defaults: Dict[str, str]


def load_litellm_models(
    config_path: str = os.path.join("src", "config", "litellm_config.yaml"),
) -> Tuple[List[str], List[str], List[str]]:
    """Load model aliases from LiteLLM config and categorize naive reader vs embedding."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        model_list = data.get("model_list", [])
        all_models = [
            m.get("model_name")
            for m in model_list
            if isinstance(m, dict) and m.get("model_name")
        ]
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
        # Deduplicate preserving order
        reader_models = list(dict.fromkeys(reader_models))
        embedding_models = list(dict.fromkeys(embedding_models))
        all_models = list(dict.fromkeys(all_models))
        return reader_models, embedding_models, all_models
    except Exception:
        # Fallback to defaults
        return (
            [DEFAULT_READER_MODEL],
            [DEFAULT_EMBEDDING_MODEL],
            [DEFAULT_READER_MODEL, DEFAULT_EMBEDDING_MODEL],
        )


def ensure_pipelines_available():
    if run_rag_clause_extraction is None or run_advanced_rag_pipeline is None:
        raise RuntimeError(
            "Pipeline modules not importable. Ensure project is run from repository root so 'src' is on PYTHONPATH."
        )


@app.get("/api/v1/health")
def health() -> Dict[str, Any]:
    lite_status = {"url": LITELLM_PROXY_URL, "ok": False}
    try:
        r = requests.get(f"{LITELLM_PROXY_URL.rstrip('/')}/v1/models", timeout=5)
        lite_status["ok"] = r.status_code < 500
        lite_status["status_code"] = r.status_code
    except Exception as e:
        lite_status["error"] = str(e)
    lf_status = {"host": LANGFUSE_HOST, "enabled": LANGFUSE_ENABLED}
    vec_status = {"backend": "chroma", "mode": "in-process"}
    return {
        "status": "ok" if lite_status.get("ok") else "degraded",
        "litellm": lite_status,
        "langfuse": lf_status,
        "vector": vec_status,
        "defaults": {
            "reader_model": DEFAULT_READER_MODEL,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "use_hyde": DEFAULT_USE_HYDE,
        },
    }


@app.get("/api/v1/models", response_model=ModelsResponse)
def list_models():
    readers, embeddings, all_models = load_litellm_models()
    return {
        "reader_models": readers,
        "embedding_models": embeddings,
        "all_models": all_models,
        "defaults": {
            "reader_model": DEFAULT_READER_MODEL,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
        },
    }


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/extract-clauses", response_model=ExtractionResponse)
def extract_clauses(req: ExtractionRequest) -> ExtractionResponse:
    ensure_pipelines_available()
    opts = req.options or ExtractionOptions()
    clause_types = req.clause_types or BASE_CLAUSE_TYPES
    timings: Dict[str, Optional[float]] = {
        "chunk_ms": None,
        "embed_ms": None,
        "retrieve_ms": None,
        "read_ms": None,
    }
    usage: Dict[str, Optional[int]] = {"input_tokens": None, "output_tokens": None}
    model_info = {
        "reader_model": opts.reader_model or DEFAULT_READER_MODEL,
        "embedding_model": opts.embedding_model or DEFAULT_EMBEDDING_MODEL,
        "use_hyde": bool(opts.use_hyde),
    }
    try:
        t0 = time.perf_counter()
        if opts.use_hyde:
            # Advanced pipeline with HyDE
            result = run_advanced_rag_pipeline(
                document_text=req.document_text,
                document_id=req.document_id,
                user_query=req.user_query,
                target_clause_types=clause_types,
                language=req.language,
                reader_model_override=model_info["reader_model"],
                embedding_model_override=model_info["embedding_model"],
            )
            t1 = time.perf_counter()
            timings["read_ms"] = (t1 - t0) * 1000.0
            RAG_STAGE_HISTOGRAM.labels("read").observe(t1 - t0)
            # Pipelines return annotations as list of dicts
            predicted = result.get("predicted_annotations", []) or []
            retrieved = result.get("retrieved_chunks", []) or []
            raw = result.get("reader_llm_output_raw")
            trace_id = result.get("langfuse_trace_id")
            # Usage not exposed directly by pipeline; keep None for now
        else:
            # Baseline RAG pipeline (no HyDE)
            result = run_rag_clause_extraction(
                document_text=req.document_text,
                document_id=req.document_id,
                language=req.language,
                reader_model_alias_param=model_info["reader_model"],
                embedding_model_alias_param=model_info["embedding_model"],
                litellm_proxy_url_param=LITELLM_PROXY_URL,
            )
            t1 = time.perf_counter()
            timings["read_ms"] = (t1 - t0) * 1000.0
            RAG_STAGE_HISTOGRAM.labels("read").observe(t1 - t0)
            predicted = result.get("predicted_annotations", []) or []
            retrieved = result.get("retrieved_chunk_texts", []) or []
            raw = None
            trace_id = result.get("rag_trace_id")
        # Map to response model
        predicted_items = [
            PredictedAnnotation(**p)
            for p in predicted
            if isinstance(p, dict) and "clause_type" in p and "clause_text" in p
        ]
        # Observe tokens if present (future)
        # if usage.get("input_tokens"): LLM_TOKENS.labels(model_info["reader_model"], "reader").inc(usage["input_tokens"] or 0)
        return ExtractionResponse(
            predicted_annotations=predicted_items,
            retrieved_chunks=retrieved,
            reader_llm_output_raw=raw,
            trace_id=trace_id,
            timings=timings,
            usage=usage,
            model_info=model_info,
        )
    except HTTPException:
        raise
    except Exception as e:
        RAG_ERRORS.labels("pipeline", e.__class__.__name__).inc()
        raise HTTPException(status_code=500, detail={"message": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )
