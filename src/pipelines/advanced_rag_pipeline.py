"""
Advanced RAG Pipeline with HyDE

Hypothetical Document Embedding (HyDE) enhanced retrieval:
  1. Generate hypothetical ideal document from query
  2. Embed hypothetical doc for better semantic matching
  3. Standard chunk → embed → retrieve → read flow

Improves retrieval quality for complex legal queries.
"""

import os
import logging
from typing import Any, Dict, List, Optional

import chromadb
import litellm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv

from src.pipelines.rag_pipeline import (
    get_embeddings,
    clean_json_block,
    validate_and_parse_json,
    CLAUSE_TYPES,
)

load_dotenv(override=True)

# --- Configuration ---
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
EMBEDDING_MODEL_ALIAS = os.getenv("EMBEDDING_MODEL_ALIAS", "text-embedding-3-small")
HYDE_MODEL_ALIAS = os.getenv("HYDE_MODEL_ALIAS", "gpt-4.1-mini")
READER_MODEL_ALIAS = os.getenv("RAG_READER_MODEL_ALIAS", "gemma3-4b-it-qat")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Global Langfuse client
try:
    langfuse_client = Langfuse()
except Exception as e:
    logger.warning(f"Langfuse init failed: {e}")
    langfuse_client = None

# --- Prompts ---
HYDE_PROMPT_TEMPLATE = """USER REQUEST: "{user_query}"

Write a short example paragraph of legal text from a European Union regulation that would answer this request.
Be specific and detailed. No explanation, just the raw legal text.

EXAMPLE PARAGRAPH:"""

READER_PROMPT_TEMPLATE = """You are an expert legal assistant. Extract clauses matching these types: [{clause_types}].

USER_QUERY: {user_query}

CONTEXT:
---
{document_context}
---

Return a valid JSON list. Each object has "clause_type" and "clause_text" keys.
If no relevant clauses found, return [].

JSON List:"""


@observe()
def run_advanced_rag_pipeline(
    document_text: str,
    document_id: str,
    user_query: str,
    target_clause_types: List[str],
    language: str = "en",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    reader_model_override: Optional[str] = None,
    hyde_model_override: Optional[str] = None,
    embedding_model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run HyDE-enhanced RAG pipeline.

    Args:
        document_text: Full document text
        document_id: Unique identifier for tracing
        user_query: User's extraction query
        target_clause_types: List of clause types to extract
        reader_model_override: Override reader LLM model
        hyde_model_override: Override HyDE generation model
        embedding_model_override: Override embedding model

    Returns:
        Dict with predicted_annotations, retrieved_chunks, hyde_document, error_message
    """
    current_reader = reader_model_override or READER_MODEL_ALIAS
    current_hyde = hyde_model_override or HYDE_MODEL_ALIAS
    current_embedding = embedding_model_override or EMBEDDING_MODEL_ALIAS

    langfuse_context.update_current_observation(
        metadata={
            "document_id": document_id,
            "user_query": user_query,
            "target_clause_types": target_clause_types,
            "reader_model": current_reader,
            "hyde_model": current_hyde,
            "embedding_model": current_embedding,
        }
    )

    result: Dict[str, Any] = {
        "document_id": document_id,
        "user_query": user_query,
        "target_clause_types": target_clause_types,
        "hyde_document": None,
        "retrieved_chunks": [],
        "reader_llm_output_raw": None,
        "predicted_annotations": [],
        "error_message": None,
        "langfuse_trace_id": langfuse_context.get_current_trace_id(),
    }

    parent_obs_id = langfuse_context.get_current_observation_id()

    # 1. HyDE: Generate hypothetical document
    hyde_span = None
    if langfuse_client and parent_obs_id:
        hyde_span = langfuse_client.generation(
            name="HyDE-Generation",
            input={"user_query": user_query},
            model=current_hyde,
            parent_observation_id=parent_obs_id,
        )

    try:
        hyde_prompt = HYDE_PROMPT_TEMPLATE.format(user_query=user_query)
        hyde_response = litellm.completion(
            model=current_hyde,
            messages=[{"role": "user", "content": hyde_prompt}],
            api_base=LITELLM_PROXY_URL,
            custom_llm_provider="openai",
            temperature=0.1,
            timeout=60,
        )

        if hyde_response.choices and hyde_response.choices[0].message.content:
            result["hyde_document"] = hyde_response.choices[0].message.content.strip()
            if hyde_span:
                hyde_span.update(
                    output=result["hyde_document"], usage=hyde_response.usage
                )
        else:
            result["hyde_document"] = user_query
            if hyde_span:
                hyde_span.update(output={"fallback": True})

    except Exception as e:
        result["hyde_document"] = user_query
        result["error_message"] = f"HyDE error: {e}"
        if hyde_span:
            hyde_span.update(output={"error": str(e)}, level="ERROR")
        logger.error(f"HyDE generation failed: {e}", exc_info=True)
    finally:
        if hyde_span:
            hyde_span.end()

    # 2. Chunking
    chunking_span = None
    if langfuse_client and parent_obs_id:
        chunking_span = langfuse_client.span(
            name="Chunking",
            input={"doc_length": len(document_text)},
            parent_observation_id=parent_obs_id,
        )

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_text(document_text)
        if chunking_span:
            chunking_span.update(output={"num_chunks": len(chunks)})
    except Exception as e:
        result["error_message"] = f"Chunking error: {e}"
        if chunking_span:
            chunking_span.update(output={"error": str(e)}, level="ERROR")
        logger.error(f"Chunking failed: {e}", exc_info=True)
        chunks = []
    finally:
        if chunking_span:
            chunking_span.end()

    if not chunks:
        result["error_message"] = result.get("error_message") or "No chunks created"
        langfuse_context.update_current_observation(output=result)
        return result

    # 3. Embedding
    embedding_span = None
    if langfuse_client and parent_obs_id:
        embedding_span = langfuse_client.span(
            name="Embedding",
            input={"num_chunks": len(chunks)},
            parent_observation_id=parent_obs_id,
        )

    try:
        query_to_embed = result["hyde_document"] or user_query
        query_embedding = get_embeddings(
            [query_to_embed], model_alias=current_embedding
        )[0]
        chunk_embeddings = get_embeddings(chunks, model_alias=current_embedding)
        if embedding_span:
            embedding_span.update(output={"status": "success"})
    except Exception as e:
        result["error_message"] = f"Embedding error: {e}"
        if embedding_span:
            embedding_span.update(output={"error": str(e)}, level="ERROR")
        logger.error(f"Embedding failed: {e}", exc_info=True)
        langfuse_context.update_current_observation(output=result)
        return result
    finally:
        if embedding_span:
            embedding_span.end()

    # 4. Retrieval (in-memory ChromaDB)
    retrieval_span = None
    if langfuse_client and parent_obs_id:
        retrieval_span = langfuse_client.span(
            name="Retrieval",
            input={"num_chunks": len(chunks)},
            parent_observation_id=parent_obs_id,
        )

    try:
        client = chromadb.Client()
        safe_id = "".join(c if c.isalnum() else "_" for c in document_id)
        collection_name = f"hyde_{safe_id}"

        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = client.get_or_create_collection(name=collection_name)
        chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        collection.add(embeddings=chunk_embeddings, documents=chunks, ids=chunk_ids)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_RETRIEVAL,
            include=["documents"],
        )
        result["retrieved_chunks"] = (
            results["documents"][0]
            if results.get("documents") and results["documents"]
            else []
        )
        if retrieval_span:
            retrieval_span.update(
                output={"num_retrieved": len(result["retrieved_chunks"])}
            )
    except Exception as e:
        result["error_message"] = f"Retrieval error: {e}"
        if retrieval_span:
            retrieval_span.update(output={"error": str(e)}, level="ERROR")
        logger.error(f"Retrieval failed: {e}", exc_info=True)
    finally:
        if retrieval_span:
            retrieval_span.end()

    # 5. Reader LLM
    reader_span = None
    if langfuse_client and parent_obs_id:
        reader_span = langfuse_client.generation(
            name="ReaderLLM",
            input={"num_chunks": len(result["retrieved_chunks"])},
            model=current_reader,
            parent_observation_id=parent_obs_id,
        )

    try:
        if not result["retrieved_chunks"]:
            result["predicted_annotations"] = []
            if reader_span:
                reader_span.update(output="No chunks to process")
        else:
            context = "\n---\n".join(result["retrieved_chunks"])
            clause_types_str = ", ".join(f'"{ct}"' for ct in target_clause_types)

            reader_prompt = READER_PROMPT_TEMPLATE.format(
                clause_types=clause_types_str,
                user_query=user_query,
                document_context=context,
            )

            reader_response = litellm.completion(
                model=current_reader,
                messages=[{"role": "user", "content": reader_prompt}],
                api_base=LITELLM_PROXY_URL,
                custom_llm_provider="openai",
                timeout=180,
            )

            if reader_response.choices and reader_response.choices[0].message.content:
                output = reader_response.choices[0].message.content
                result["reader_llm_output_raw"] = output
                parsed = validate_and_parse_json(output, document_id)
                result["predicted_annotations"] = parsed if parsed else []
                if reader_span:
                    reader_span.update(output=output, usage=reader_response.usage)
            else:
                result["predicted_annotations"] = []
                if reader_span:
                    reader_span.update(output="No content from LLM")

    except Exception as e:
        result["error_message"] = f"Reader error: {e}"
        if reader_span:
            reader_span.update(output={"error": str(e)}, level="ERROR")
        logger.error(f"Reader LLM failed: {e}", exc_info=True)
    finally:
        if reader_span:
            reader_span.end()

    langfuse_context.update_current_observation(output=result)
    return result
