# LLM Evaluation and Benchmarks

## Why LLM Evaluation is Hard

Evaluating large language models is significantly harder than evaluating traditional ML models. There is no single scalar metric like "accuracy" that captures response quality. Responses are open-ended, contextual, and multi-dimensional — the same answer might be correct, readable, and safe but verbosely formatted.

Key evaluation dimensions:
- **Correctness:** Is the factual content accurate?
- **Faithfulness (for RAG):** Is the answer grounded in the provided context?
- **Relevance:** Does the answer address the question?
- **Completeness:** Are all aspects of the question addressed?
- **Fluency:** Is the response grammatically correct and readable?
- **Safety:** Does the response avoid harmful, biased, or offensive content?

## Standard NLP Benchmarks

### MMLU (Massive Multitask Language Understanding)
A multiple-choice benchmark covering 57 academic subjects from elementary school to professional level (law, medicine, history, math, etc.). Tests factual knowledge breadth. Top models (GPT-4, Claude 3 Opus) exceed 86% accuracy.

### HumanEval
A code generation benchmark with 164 Python programming problems. The model must write a function given a docstring; correctness is evaluated by running unit tests. Measures pass@k (the probability that at least one of k generated solutions passes all tests).

### GSM8K (Grade School Math)
800 grade-school math word problems requiring multi-step reasoning. Used to measure chain-of-thought reasoning capability. GPT-4 achieves ~92%; smaller models struggle without CoT.

### HellaSwag
A commonsense reasoning benchmark where the model must complete a sentence from 4 choices. Tests world knowledge and common sense.

### TruthfulQA
Measures whether models give truthful answers to questions that humans often answer incorrectly (due to misconceptions). Tests calibration and resistance to sycophancy.

### BEIR (Benchmarking IR)
A heterogeneous benchmark for information retrieval across 18 datasets covering different domains and task types. Used to evaluate embedding models and retrieval systems.

### MTEB (Massive Text Embedding Benchmark)
Covers 56 datasets across 8 task types (classification, clustering, retrieval, etc.). The standard benchmark for embedding model quality. all-MiniLM-L6-v2 achieves 56.3 average score; text-embedding-3-large achieves 64.6.

## RAG-Specific Evaluation

### Metrics

**Context Precision:** What fraction of retrieved chunks are actually relevant to the query? Measures retrieval precision.

**Context Recall:** What fraction of relevant information needed to answer the query is present in the retrieved chunks? Measures retrieval recall.

**Faithfulness:** Is every claim in the generated answer supported by the retrieved context? Measures hallucination rate. Computed by checking each sentence of the answer against the retrieved chunks.

**Answer Relevance:** Does the answer actually address the question asked? A faithful answer might not be relevant if it discusses tangential context.

### RAGAS Framework

RAGAS is an open-source framework for reference-free RAG evaluation. It uses LLMs themselves to evaluate RAG system outputs. Metrics include faithfulness, answer relevance, context precision, context recall, and harmlessness.

**Faithfulness computation:**
1. Break the generated answer into claims (e.g., using an LLM)
2. For each claim, check if it can be inferred from the retrieved context
3. Faithfulness = (supported claims) / (total claims)

### Hallucination Detection

Hallucination is when a model generates content not supported by the provided context or factual knowledge. In RAG systems:

**Types:**
- **Intrinsic hallucination:** Generated text contradicts the source context
- **Extrinsic hallucination:** Generated text cannot be verified from source context

**Detection methods:**
- NLI (Natural Language Inference) models to check entailment between answer and context
- Sentence-level factual consistency scoring (BERTScore, SummaC)
- LLM-as-judge: Ask a separate LLM to evaluate if each claim is supported
- Self-consistency: Generate multiple answers and check agreement

## LLM-as-Judge

A common evaluation technique uses a strong LLM (e.g., GPT-4 or Claude) to evaluate responses. The judge model is given the question, reference answer (if available), and candidate response, and rates it on a scale (1-10) or makes a binary judgement.

LLM-as-judge is cost-effective and correlates well with human judgements (inter-annotator agreement ~80%). Key risks: position bias (judges prefer the first option), verbosity bias (longer responses rated higher), self-serving bias (a model judging its own outputs).

## Human Evaluation

For high-stakes applications, human evaluation remains the gold standard. Common approaches:

**Pairwise comparison (A/B testing):** Annotators choose between two responses. Reduces cognitive load vs absolute rating.

**Likert scales:** Rate dimensions like helpfulness, safety, accuracy on 1-5 scales.

**Red teaming:** Adversarial evaluation where human testers actively try to elicit harmful outputs.

**MT-Bench:** A standardised multi-turn conversation benchmark scored by GPT-4 as judge, with results that correlate well with human preferences.

## Calibration and Uncertainty

A well-calibrated model's confidence should match its accuracy. A model saying "I'm 90% confident" should be right 90% of the time. LLMs are poorly calibrated by default — they often express high confidence when wrong (overconfident).

**Calibration metrics:** Expected Calibration Error (ECE), reliability diagrams.

**Improving calibration:**
- RLHF training with calibration objectives
- Adding uncertainty language ("I'm not certain but...", "You may want to verify...")
- Temperature scaling post-hoc
- Abstaining when below a confidence threshold

## Performance vs Cost Tradeoffs

| Model | Cost (input) | Latency (TTFT) | Quality |
|-------|-------------|----------------|---------|
| Claude Haiku | $0.25/MTok | ~200ms | Good |
| Claude Sonnet | $3/MTok | ~400ms | Very Good |
| Claude Opus | $15/MTok | ~800ms | Excellent |
| GPT-4o-mini | $0.15/MTok | ~200ms | Good |
| GPT-4o | $2.5/MTok | ~500ms | Very Good |

For most RAG Q&A workloads, Haiku or GPT-4o-mini offers the best cost-to-quality ratio. Upgrade to Sonnet/Opus only for complex reasoning tasks.

## Continuous Evaluation

Production LLM systems require continuous evaluation:
- **A/B testing:** Compare model versions on live traffic
- **Shadow evaluation:** Run new model on traffic without serving it; compare offline
- **Automated regression suites:** Run a curated set of test cases on every model/prompt change
- **Monitoring:** Track query distribution shifts, refusal rates, user satisfaction (thumbs up/down)
