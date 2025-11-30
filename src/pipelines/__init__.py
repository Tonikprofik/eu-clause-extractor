"""
EU Contract Analyzer - Pipeline Modules

Three extraction pipelines demonstrating LLMOps patterns:
  - rag_pipeline: Standard RAG (chunk → embed → retrieve → read)
  - agentic_extractor: RAG + LangGraph critique/refine loop
  - advanced_rag_pipeline: HyDE-enhanced retrieval

All pipelines are traced via Langfuse for observability.
"""

from src.pipelines.rag_pipeline import (
    run_rag_clause_extraction,
    CLAUSE_TYPES,
    SCHEMA,
    clean_json_block,
    validate_and_parse_json,
    get_embeddings,
)

from src.pipelines.advanced_rag_pipeline import run_advanced_rag_pipeline

# Agentic pipeline requires async - import separately when needed
# from src.pipelines.agentic_extractor import run_agentic_pipeline

__all__ = [
    "run_rag_clause_extraction",
    "run_advanced_rag_pipeline",
    "CLAUSE_TYPES",
    "SCHEMA",
    "clean_json_block",
    "validate_and_parse_json",
    "get_embeddings",
]
