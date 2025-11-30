"""
RAG Pipeline for EU Clause Extraction

Standard Retrieval-Augmented Generation pipeline:
  1. Chunk document text
  2. Embed chunks via LiteLLM proxy
  3. Store in ChromaDB vector store
  4. Retrieve top-k relevant chunks
  5. Reader LLM extracts clauses from context

Traced via Langfuse @observe decorator.
"""

import json
import os
import re
import logging
from typing import Any, Dict, List, Optional

import chromadb
import litellm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse.decorators import observe, langfuse_context
from langfuse.openai import OpenAI as LangfuseOpenAIClient
from jsonschema import validate, ValidationError
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Configuration ---
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
EMBEDDING_MODEL_ALIAS = os.getenv("EMBEDDING_MODEL_ALIAS", "text-embedding-3-small")
READER_MODEL_ALIAS = os.getenv("RAG_READER_MODEL_ALIAS", "claude-3-7")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
CHROMA_RAG_PATH = os.getenv("CHROMA_RAG_PATH", "./chroma_db_poc2_rag_en")
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "poc2_rag_en_clauses")
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# --- Logging ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

# --- Clause Types & Schema ---
CLAUSE_TYPES = [
    "Subject Matter & Scope",
    "Definitions",
    "Obligations of Member States",
    "Penalties",
    "Entry into Force & Application",
]

SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "clause_type": {"type": "string", "enum": CLAUSE_TYPES},
            "clause_text": {"type": "string", "minLength": 10},
        },
        "required": ["clause_type", "clause_text"],
        "additionalProperties": False,
    },
}

# --- Prompts ---
READER_SYSTEM_PROMPT = (
    "You are an expert legal assistant. Based on the provided CONTEXT from a larger document, "
    "extract all clauses that match any of the following types: "
    f"{', '.join(CLAUSE_TYPES)}. "
    "Return your findings as a valid JSON list of objects. Each object must have exactly two keys: "
    "'clause_type' (one of the authorized types) and 'clause_text' (the full extracted text of the clause). "
    "If no relevant clauses are found in the context, return an empty JSON list []."
)

READER_USER_PROMPT_TEMPLATE = (
    "CONTEXT:\n---\n{context_str}\n---\nJSON list of extracted clauses:"
)


def clean_json_block(raw_output: str) -> Optional[str]:
    """Extract JSON from markdown code blocks or raw text."""
    if not raw_output or not raw_output.strip():
        return None

    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_output, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        # Find JSON structure in raw text
        start_brace = raw_output.find("{")
        start_bracket = raw_output.find("[")

        if start_brace == -1 and start_bracket == -1:
            return None

        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            json_start_index = start_brace
            end_char = "}"
        else:
            json_start_index = start_bracket
            end_char = "]"

        json_end_index = raw_output.rfind(end_char)
        if json_start_index != -1 and json_end_index > json_start_index:
            cleaned = raw_output[json_start_index : json_end_index + 1].strip()
        else:
            return None

    # Validate structure
    if (cleaned.startswith("{") and cleaned.endswith("}")) or (
        cleaned.startswith("[") and cleaned.endswith("]")
    ):
        return cleaned
    return None


def validate_and_parse_json(
    raw_output: str,
    document_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """Clean, parse, and validate JSON output against schema."""
    cleaned_output = clean_json_block(raw_output)
    if not cleaned_output:
        logger.error(f"Validation Error ({document_id}): Empty after cleaning.")
        return None

    try:
        parsed_data = json.loads(cleaned_output)
    except json.JSONDecodeError as e:
        logger.error(f"Validation Error ({document_id}): JSON parse failed: {e}")
        return None

    try:
        validate(instance=parsed_data, schema=SCHEMA)
        if isinstance(parsed_data, list):
            validated_items = []
            for item in parsed_data:
                if (
                    isinstance(item, dict)
                    and item.get("clause_type") in CLAUSE_TYPES
                    and isinstance(item.get("clause_text"), str)
                    and len(item["clause_text"].strip()) >= 10
                ):
                    item["clause_text"] = item["clause_text"].strip()
                    validated_items.append(item)
            return validated_items
        return None
    except ValidationError as e:
        logger.error(f"Schema Error ({document_id}): {e.message}")
        return None


def get_rag_collection(collection_name: str = RAG_COLLECTION_NAME):
    """Get or create ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_RAG_PATH)
    return client.get_or_create_collection(name=collection_name)


def get_embeddings(
    texts: List[str], model_alias: str = EMBEDDING_MODEL_ALIAS
) -> List[List[float]]:
    """Get embeddings via LiteLLM."""
    try:
        response = litellm.embedding(model=model_alias, input=texts)
        return [item["embedding"] for item in response.data]
    except Exception as e:
        logger.error(f"Embedding error ({model_alias}): {e}", exc_info=True)
        raise


@observe()
def _rag_perform_chunking(
    document_text: str,
    document_id: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[List[str]]:
    """Chunk document text."""
    langfuse_context.update_current_trace(
        metadata={
            "document_id": document_id,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        session_id=session_id,
        user_id=user_id,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(document_text)
    if not chunks:
        logger.warning(f"No chunks created for {document_id}")
        return None
    langfuse_context.update_current_trace(output={"num_chunks": len(chunks)})
    return chunks


@observe()
def _rag_perform_embedding(
    chunks: List[str],
    document_id: str,
    embedding_model_alias_param: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[List[List[float]]]:
    """Embed text chunks."""
    langfuse_context.update_current_trace(
        metadata={
            "document_id": document_id,
            "num_chunks": len(chunks),
            "embedding_model": embedding_model_alias_param,
        },
        session_id=session_id,
        user_id=user_id,
    )
    try:
        embeddings = get_embeddings(chunks, model_alias=embedding_model_alias_param)
        langfuse_context.update_current_trace(output={"status": "success"})
        return embeddings
    except Exception as e:
        logger.error(f"Embedding failed for {document_id}: {e}", exc_info=True)
        langfuse_context.update_current_trace(
            output={"status": "failed", "error": str(e)}
        )
        return None


@observe()
def run_rag_clause_extraction(
    document_text: str,
    document_id: str,
    language: str = "en",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    reader_model_alias_param: Optional[str] = None,
    litellm_proxy_url_param: Optional[str] = None,
    embedding_model_alias_param: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full RAG clause extraction pipeline.

    Args:
        document_text: Full document text to extract clauses from
        document_id: Unique identifier for tracing
        language: Document language (default: "en")
        reader_model_alias_param: Override reader LLM model
        embedding_model_alias_param: Override embedding model
        litellm_proxy_url_param: Override LiteLLM proxy URL

    Returns:
        Dict with predicted_annotations, rag_trace_id, retrieved_chunk_texts, error
    """
    current_reader_model = reader_model_alias_param or READER_MODEL_ALIAS
    current_embedding_model = embedding_model_alias_param or EMBEDDING_MODEL_ALIAS
    current_proxy_url = litellm_proxy_url_param or LITELLM_PROXY_URL

    rag_trace_id = langfuse_context.get_current_trace_id()
    langfuse_context.update_current_trace(
        name=f"RAG_ClauseExtraction-{current_reader_model}",
        metadata={
            "document_id": document_id,
            "language": language,
            "reader_model": current_reader_model,
            "embedding_model": current_embedding_model,
            "top_k": TOP_K_RETRIEVAL,
        },
        session_id=session_id,
        user_id=user_id,
    )

    # 1. Chunking
    chunks = _rag_perform_chunking(document_text, document_id, session_id, user_id)
    if chunks is None:
        return {
            "predicted_annotations": [],
            "rag_trace_id": rag_trace_id,
            "retrieved_chunk_texts": [],
            "error": "Chunking failed",
        }

    # 2. Embedding
    chunk_embeddings = _rag_perform_embedding(
        chunks, document_id, current_embedding_model, session_id, user_id
    )
    if chunk_embeddings is None:
        return {
            "predicted_annotations": [],
            "rag_trace_id": rag_trace_id,
            "retrieved_chunk_texts": chunks,
            "error": "Embedding failed",
        }

    # 3. Vector Store
    collection = get_rag_collection(RAG_COLLECTION_NAME)
    chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    collection.add(embeddings=chunk_embeddings, documents=chunks, ids=chunk_ids)

    # 4. Retrieval
    query_text = f"Extract legal clauses: {', '.join(CLAUSE_TYPES)}"
    try:
        query_embedding = get_embeddings(
            [query_text], model_alias=current_embedding_model
        )[0]
    except Exception as e:
        return {
            "predicted_annotations": [],
            "rag_trace_id": rag_trace_id,
            "retrieved_chunk_texts": [],
            "error": f"Query embedding failed: {e}",
        }

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K_RETRIEVAL,
        include=["documents"],
    )
    retrieved_docs = (
        results["documents"][0]
        if results.get("documents") and results["documents"]
        else []
    )

    # 5. Reader LLM
    predicted_annotations = []
    if retrieved_docs:
        context_str = "\n\n---\n\n".join(retrieved_docs)
        reader_client = LangfuseOpenAIClient(
            base_url=current_proxy_url,
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
            timeout=120,
        )
        messages = [
            {"role": "system", "content": READER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": READER_USER_PROMPT_TEMPLATE.format(context_str=context_str),
            },
        ]
        try:
            completion = reader_client.chat.completions.create(
                model=current_reader_model,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
                name=f"RAG-Reader-{document_id}",
                session_id=session_id,
                user_id=user_id,
            )
            output = completion.choices[0].message.content
            parsed = validate_and_parse_json(output, document_id)
            if parsed:
                predicted_annotations = parsed
        except Exception as e:
            logger.error(f"Reader LLM error for {document_id}: {e}", exc_info=True)

    langfuse_context.update_current_trace(
        output={
            "annotations_count": len(predicted_annotations),
            "chunks_retrieved": len(retrieved_docs),
        }
    )

    return {
        "predicted_annotations": predicted_annotations,
        "rag_trace_id": rag_trace_id,
        "retrieved_chunk_texts": retrieved_docs,
    }
