"""
Agentic Clause Extractor with LangGraph

Multi-step agent pipeline:
  1. RAG extraction (reuses rag_pipeline)
  2. Critique & refine via separate LLM

Uses LangGraph for state management and Langfuse for observability.
"""

import asyncio
import os
import json
import logging
from typing import TypedDict, List, Dict, Any, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langfuse.decorators import observe, langfuse_context
from langfuse.callback import CallbackHandler
from litellm import acompletion as litellm_acompletion
from jsonschema import validate, ValidationError

from src.pipelines.rag_pipeline import run_rag_clause_extraction, clean_json_block

load_dotenv(override=True)

# --- Configuration ---
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
AGENT_RAG_MODEL_ALIAS = os.getenv("RAG_READER_MODEL_ALIAS", "gemma3-4b-it-qat")
AGENT_CRITIQUE_MODEL_ALIAS = os.getenv("LLM_CRITIQUE_MODEL_ALIAS", "gemma3-4b-it-qat")
AGENT_EMBEDDING_MODEL_ALIAS = os.getenv(
    "EMBEDDING_MODEL_ALIAS", "text-embedding-3-small"
)

logger = logging.getLogger(__name__)

# --- Schema for Critique Output ---
CRITIQUE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "critique_text": {"type": "string"},
        "revised_clauses_json": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clause_type": {"type": "string", "minLength": 1},
                    "clause_text": {"type": "string", "minLength": 1},
                },
                "required": ["clause_type", "clause_text"],
            },
        },
    },
    "required": ["critique_text", "revised_clauses_json"],
}


class AgentState(TypedDict):
    """State passed through the agent graph."""

    document_text: str
    document_id: str
    extracted_clauses_raw: Optional[List[Dict[str, Any]]]
    critique_text: Optional[str]
    final_clauses: Optional[List[Dict[str, Any]]]
    error_message: Optional[str]


async def rag_extract_node(state: AgentState) -> AgentState:
    """Node: Extract clauses using RAG pipeline."""
    doc_id = state["document_id"]
    logger.info(f"Agent: RAG extraction for {doc_id}")

    try:
        loop = asyncio.get_event_loop()
        rag_output = await loop.run_in_executor(
            None,
            run_rag_clause_extraction,
            state["document_text"],
            doc_id,
            "en",
            None,
            None,
            AGENT_RAG_MODEL_ALIAS,
            LITELLM_PROXY_URL,
            AGENT_EMBEDDING_MODEL_ALIAS,
        )

        if rag_output.get("error"):
            return {
                **state,
                "error_message": str(rag_output["error"]),
                "extracted_clauses_raw": None,
            }

        return {
            **state,
            "extracted_clauses_raw": rag_output.get("predicted_annotations"),
            "error_message": None,
        }
    except Exception as e:
        logger.exception(f"RAG extraction failed for {doc_id}")
        return {
            **state,
            "error_message": f"RAG error: {e}",
            "extracted_clauses_raw": None,
        }


async def critique_refine_node(state: AgentState) -> AgentState:
    """Node: Critique and refine extracted clauses."""
    doc_id = state["document_id"]
    logger.info(f"Agent: Critique/refine for {doc_id}")

    if state.get("error_message"):
        return state

    if not state.get("extracted_clauses_raw"):
        return {
            **state,
            "final_clauses": [],
            "critique_text": "No raw clauses to critique.",
        }

    document_snippet = state["document_text"][:2000]
    raw_clauses_json = json.dumps(state["extracted_clauses_raw"], indent=2)

    critique_prompt = f"""You are a legal expert reviewing machine-extracted clauses.

Document Snippet:
---
{document_snippet}
---

Machine Extracted Clauses:
---
{raw_clauses_json}
---

Tasks:
1. Briefly critique the quality and completeness of the extracted clauses.
2. Provide a revised list in valid JSON format.

Output as JSON with keys: "critique_text" (string) and "revised_clauses_json" (list of clause objects).
Each clause object has "clause_type" and "clause_text" keys.
"""

    messages = [
        {"role": "system", "content": "You are an expert legal assistant."},
        {"role": "user", "content": critique_prompt},
    ]

    try:
        response = await litellm_acompletion(
            model=AGENT_CRITIQUE_MODEL_ALIAS,
            messages=messages,
            api_base=LITELLM_PROXY_URL,
            custom_llm_provider="openai",
        )

        llm_output = response.choices[0].message.content
        cleaned = clean_json_block(llm_output)

        if not cleaned:
            return {
                **state,
                "error_message": "Critique output empty after cleaning",
                "critique_text": "Error: empty output",
                "final_clauses": state["extracted_clauses_raw"],
            }

        parsed = json.loads(cleaned)
        validate(instance=parsed, schema=CRITIQUE_OUTPUT_SCHEMA)

        return {
            **state,
            "critique_text": parsed.get("critique_text", ""),
            "final_clauses": parsed.get("revised_clauses_json", []),
            "error_message": None,
        }

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Critique parse/validation error for {doc_id}: {e}")
        return {
            **state,
            "error_message": f"Critique error: {e}",
            "critique_text": "Parse error",
            "final_clauses": state["extracted_clauses_raw"],
        }
    except Exception as e:
        logger.exception(f"Critique node failed for {doc_id}")
        return {
            **state,
            "error_message": f"Critique exception: {e}",
            "critique_text": "Exception occurred",
            "final_clauses": state["extracted_clauses_raw"],
        }


# --- Build LangGraph ---
workflow = StateGraph(AgentState)
workflow.add_node("rag_extract", rag_extract_node)
workflow.add_node("critique_refine", critique_refine_node)
workflow.set_entry_point("rag_extract")
workflow.add_edge("rag_extract", "critique_refine")
workflow.add_edge("critique_refine", END)

agent_app = workflow.compile()


@observe(name="AgenticClauseExtraction_Pipeline")
async def run_agentic_pipeline(document_text: str, document_id: str) -> Dict[str, Any]:
    """
    Run the full agentic extraction pipeline.

    Args:
        document_text: Full document to extract clauses from
        document_id: Unique identifier for tracing

    Returns:
        Final state dict with final_clauses, critique_text, error_message
    """
    langfuse_handler = langfuse_context.get_current_langchain_handler()

    initial_state: AgentState = {
        "document_text": document_text,
        "document_id": document_id,
        "extracted_clauses_raw": None,
        "critique_text": None,
        "final_clauses": None,
        "error_message": None,
    }

    try:
        result = await agent_app.ainvoke(
            initial_state, config={"callbacks": [langfuse_handler]}
        )
        final_state = {
            "document_id": result.get("document_id", document_id),
            "extracted_clauses_raw": result.get("extracted_clauses_raw"),
            "critique_text": result.get("critique_text"),
            "final_clauses": result.get("final_clauses"),
            "error_message": result.get("error_message"),
        }
    except Exception as e:
        logger.exception(f"Agent graph error for {document_id}")
        final_state = {
            "document_id": document_id,
            "extracted_clauses_raw": None,
            "critique_text": "Agent failed",
            "final_clauses": None,
            "error_message": f"Graph error: {e}",
        }

    langfuse_context.update_current_trace(
        output={
            "final_clauses_count": len(final_state.get("final_clauses") or []),
            "has_critique": bool(final_state.get("critique_text")),
            "error": final_state.get("error_message"),
        },
        metadata={
            "rag_model": AGENT_RAG_MODEL_ALIAS,
            "critique_model": AGENT_CRITIQUE_MODEL_ALIAS,
        },
    )

    return final_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def test():
        sample = "Article 1: Definitions. 'Widget' means a device. Article 2: Obligations. Payment due in 30 days."
        result = await run_agentic_pipeline(sample, "test_agent_001")
        print(f"Final clauses: {len(result.get('final_clauses') or [])}")
        print(f"Critique: {result.get('critique_text')}")

    asyncio.run(test())
