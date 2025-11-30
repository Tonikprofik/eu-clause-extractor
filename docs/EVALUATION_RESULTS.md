# Evaluation Results

Comprehensive evaluation of clause extraction pipelines on the EU regulations gold dataset.

## Dataset

| Property | Value |
|----------|-------|
| **Gold Set Size** | 27 documents |
| **Language** | English (EUR-Lex EN) |
| **Document Types** | Regulations, Decisions |
| **Annotation Source** | LLM-assisted + human review |

> ⚠️ **Disclaimer**: The gold annotations were created through LLM-assisted annotation followed by human review. They represent a reasonable ground truth for comparative evaluation but may not capture all valid clause interpretations. Use for relative pipeline comparison rather than absolute quality measurement.

## Pipeline Comparison

### Main Results

| Pipeline | F1 | Precision | Recall | Success Rate |
|----------|---:|----------:|-------:|-------------:|
| **Baseline** (Claude 3.5, Direct) | 0.893 | 0.895 | 0.895 | 27/27 |
| **RAG** (Gemma-3, K=3) | 0.735 | 0.611 | 0.963 | 27/27 |
| **RAG** (GPT-4.1-mini, K=5) | 0.686 | 0.654 | 0.750 | 27/27 |
| **RAG** (Claude 3.7, K=5) | 0.686 | 0.654 | 0.750 | 27/27 |
| **Agent** (Gemma3 + Gemma3 Critique) | 0.430 | 0.509 | 0.432 | 27/27 |
| **Agent** (Gemma3 + Claude 3.7 Critique) | 0.164 | 0.138 | 0.216 | 27/27 |

### Key Observations

1. **Baseline wins on F1**: Direct extraction with full document context outperforms chunked retrieval approaches
2. **RAG maximizes recall**: Gemma-3 K=3 achieves 0.963 recall — nearly all relevant clauses are surfaced
3. **Precision-recall tradeoff**: Higher K values don't consistently improve results; K=3 often outperforms K=5
4. **Agent overhead**: Self-critique loops add complexity without corresponding accuracy gains on this dataset

## RAG Configuration Sweep

Systematic evaluation across retrieval parameters and reader models.

| Configuration | F1 | Precision | Recall | Retrieval Recall |
|---------------|---:|----------:|-------:|-----------------:|
| GPT-4.1, K=3 | 0.698 | 0.641 | 0.830 | 0.173 |
| GPT-4.1-mini, K=3 | 0.686 | 0.654 | 0.750 | 0.093 |
| GPT-4.1-mini, K=5 | 0.686 | 0.654 | 0.750 | 0.136 |
| Gemma-3, K=3 | **0.735** | 0.611 | **0.963** | 0.093 |
| Gemma-3, K=5 | 0.614 | 0.519 | 0.787 | 0.136 |
| Claude 3.7, K=3 | 0.686 | 0.654 | 0.750 | 0.093 |
| Claude 3.7, K=5 | 0.686 | 0.654 | 0.750 | 0.136 |

### Retrieval Recall Note

Retrieval recall measures how many gold clause types appear in the retrieved chunks. Low values (0.09-0.17) indicate that the chunking strategy may not align well with clause boundaries in EU regulations.

## LLM-as-Judge Evaluation

Using different LLMs to score extraction quality (1-10 scale normalized to 0-1).

| Reader Model | Judge Model | Judge Score | Notes |
|--------------|-------------|------------:|-------|
| GPT-4.1-mini | Claude 3.5 | 0.269 | Stricter scoring |
| GPT-4.1-mini | GPT-4.1-mini | 0.091 | Self-evaluation bias |
| GPT-4.1-mini | Gemma3 | 0.763 | More lenient |
| Gemma-3 | Gemma3 | 0.556 | Moderate |

### Insights

- **Model-specific bias**: LLMs tend to score their own outputs higher (GPT judging GPT = 0.091 vs Claude judging GPT = 0.269)
- **Gemma3 is lenient**: Consistently scores higher than other judges
- **Claude is strict**: Most critical evaluations, potentially more aligned with human expectations

## Agent Pipeline Results

Comparison of RAG + Critique architectures.

| RAG Model | Critique Model | F1 | Judge Score | Critique Behavior |
|-----------|----------------|---:|------------:|-------------------|
| Gemma3 | Gemma3 | 0.430 | 0.683 | Conservative refinement |
| Gemma3 | Claude 3.7 | 0.164 | 0.376 | Aggressive filtering |
| GPT-4.1-mini | Claude 3.7 | 0.247 | 0.347 | Over-correction |

### Why Agents Underperform

1. **Over-critique**: Strong critique models (Claude 3.7) remove valid clauses, reducing recall
2. **Context loss**: The refinement loop loses document structure information
3. **Error propagation**: Initial RAG errors compound through multiple iterations

## Recommendations

### For Production Use

1. **Start with Baseline**: If documents fit in context window, direct extraction is most accurate
2. **Use RAG for recall-critical tasks**: When missing clauses is costly, RAG with K=3 captures more
3. **Skip agents for simple regulations**: Overhead isn't justified for shorter EU documents

### For Future Research

1. **Improve chunking**: Clause-aware chunking that respects legal document structure
2. **Hybrid approaches**: Baseline for short docs, RAG for long docs
3. **Selective critique**: Apply agent refinement only to low-confidence extractions

## Reproducibility

All evaluation runs are logged in Langfuse with full traces. Result files:

```
baseline_evaluation_results/
rag_evaluation_results/
agent_evaluation_results/
adv_rag_evaluation_results/
```

To reproduce:

```bash
# Run RAG evaluation
python src/poc2_run_evaluation.py

# Run Agent evaluation  
python src/poc3_run_agent_eval.py

# Analyze results
python src/poc3_analyze_agents.py
```
