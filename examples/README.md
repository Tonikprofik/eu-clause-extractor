# Examples & Cookbooks

Notebooks demonstrating core integrations and patterns used in the EU Contract Analyzer.

## Project Notebooks

| Notebook | Description |
|----------|-------------|
| [analysis_results.ipynb](./analysis_results.ipynb) | Evaluation analysis comparing Baseline → RAG → Agentic pipelines |

## Langfuse Cookbooks

Reference notebooks from [Langfuse's official cookbook](https://github.com/langfuse/langfuse-docs) demonstrating patterns used in this project.

### Core Integrations

| Notebook | Demonstrates |
|----------|--------------|
| [integration_langgraph.ipynb](./integration_langgraph.ipynb) | LangGraph + Langfuse tracing (agentic pipeline) |
| [integration_litellm_proxy.ipynb](./integration_litellm_proxy.ipynb) | LiteLLM proxy for multi-provider routing |
| [Vectordb_with_chroma.ipynb](./Vectordb_with_chroma.ipynb) | ChromaDB vector store patterns |
| [python_decorators.ipynb](./python_decorators.ipynb) | `@observe()` decorator for tracing |

### Evaluation & Datasets

| Notebook | Demonstrates |
|----------|--------------|
| [langfuseexample_external_evaluation_pipelines.ipynb](./langfuseexample_external_evaluation_pipelines.ipynb) | LLM-as-Judge evaluation pipelines |
| [evaluation_of_rag_with_ragas.ipynb](./evaluation_of_rag_with_ragas.ipynb) | RAGAS metrics for RAG evaluation |
| [datasets.ipynb](./datasets.ipynb) | Dataset management in Langfuse |
| [langfuseexample_synthetic_datasets.ipynb](./langfuseexample_synthetic_datasets.ipynb) | Synthetic dataset generation |

### Additional Integrations

| Notebook | Demonstrates |
|----------|--------------|
| [integration_google_vertex_and_gemini.ipynb](./integration_google_vertex_and_gemini.ipynb) | Google Vertex AI / Gemini |
| [integration_gradio_chatbot.ipynb](./integration_gradio_chatbot.ipynb) | Gradio UI integration |
| [otel_integration_mlflow.ipynb](./otel_integration_mlflow.ipynb) | OpenTelemetry + MLflow |
| [js_langfuse_sdk.ipynb](./js_langfuse_sdk.ipynb) | JavaScript/TypeScript SDK |

## Quick Start

```bash
# Install dependencies
pip install -r ../requirements.txt

# Set environment variables
export LANGFUSE_PUBLIC_KEY=pk-...
export LANGFUSE_SECRET_KEY=sk-...
export LITELLM_PROXY_URL=http://localhost:4000

# Open in Jupyter
jupyter lab examples/
```
