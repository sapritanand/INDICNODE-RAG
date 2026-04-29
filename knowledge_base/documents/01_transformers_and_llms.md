# Transformer Architecture and Large Language Models

## What is the Transformer Architecture?

The Transformer is a deep learning architecture introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. It replaced recurrent neural networks (RNNs) and LSTMs as the dominant architecture for sequence modelling tasks. The core innovation is the **self-attention mechanism**, which allows each token in a sequence to attend to every other token simultaneously, rather than processing tokens one at a time.

A Transformer consists of two main components: an **encoder** (which processes the input and creates contextual representations) and a **decoder** (which generates the output sequence). Models like BERT use only the encoder, while GPT-style models use only the decoder. Models like T5 use both.

## Self-Attention Mechanism

Self-attention computes three vectors for each token: Query (Q), Key (K), and Value (V). The attention score between two tokens is computed as the dot product of Q and K, scaled by the square root of the key dimension, then passed through a softmax function. These scores are used to compute a weighted sum of the Value vectors. This allows the model to capture long-range dependencies without the vanishing gradient problems of RNNs.

Multi-head attention runs several attention heads in parallel, each learning to attend to different aspects of the input (e.g., syntactic relationships, coreference, semantic similarity). The outputs of all heads are concatenated and projected.

## Positional Encoding

Since Transformers process all tokens in parallel (unlike RNNs), they have no inherent notion of order. Positional encodings are added to token embeddings to inject sequence information. Original Transformers used fixed sinusoidal encodings. Modern LLMs often use learned positional embeddings or rotary position embeddings (RoPE), which generalise better to long sequences.

## What are Large Language Models (LLMs)?

Large Language Models are Transformer-based models trained on massive text corpora using a **language modelling objective** — typically next-token prediction (autoregressive LMs like GPT) or masked token prediction (masked LMs like BERT). "Large" refers both to parameter count (billions to trillions) and training data scale (hundreds of billions to trillions of tokens).

Examples include:
- **GPT-4** (OpenAI) — decoder-only, ~1.8T parameters estimated
- **Claude** (Anthropic) — decoder-only, Constitutional AI alignment
- **LLaMA 3** (Meta) — open-weights, decoder-only, 8B–70B parameters
- **Gemini** (Google) — multimodal, encoder-decoder hybrid

## Pre-training and Fine-tuning

LLMs go through two main training phases:

**Pre-training:** The model is trained on a large, diverse corpus (Common Crawl, books, code, Wikipedia). This teaches general language understanding and world knowledge. Pre-training is computationally expensive — GPT-3 required ~3.14×10²³ FLOPs.

**Fine-tuning:** The pre-trained model is further trained on task-specific data to align it with particular use cases. Instruction fine-tuning (e.g., using datasets of instruction-response pairs) teaches the model to follow natural language instructions. RLHF (Reinforcement Learning from Human Feedback) uses human preference data to further align outputs with human values.

## Tokenisation

LLMs operate on tokens, not raw characters or words. Modern LLMs use Byte-Pair Encoding (BPE) or SentencePiece tokenisation, which splits text into subword units. A typical English word is 1–2 tokens; code tends to be more token-dense. The vocabulary size is typically 32k–100k tokens. Token limits (context window) constrain how much text the model can process at once.

## Context Window and Attention Complexity

The context window is the maximum number of tokens the model can process in a single forward pass. Standard Transformers have O(n²) attention complexity, making very long contexts expensive. Modern techniques like sliding window attention (Mistral), ring attention, and linear attention approximations extend context length efficiently. GPT-4 Turbo supports 128k tokens; Claude 3.5 supports 200k tokens.

## Emergent Capabilities

At sufficient scale, LLMs exhibit emergent capabilities — abilities not present in smaller models. These include few-shot in-context learning, chain-of-thought reasoning, code generation, and mathematical problem solving. Emergent capabilities arise unpredictably at certain parameter-count thresholds, which is not fully understood theoretically.

## Limitations of LLMs

- **Hallucination:** LLMs can generate plausible but factually incorrect content, especially for specific facts, dates, citations, and numerical reasoning.
- **Knowledge cutoff:** Training data has a cutoff date; models lack awareness of events after that date without retrieval augmentation.
- **Context window limits:** Cannot process documents longer than the context window in a single pass.
- **Reasoning limits:** Struggle with multi-step mathematical reasoning without chain-of-thought scaffolding.
- **Bias:** Trained on human-generated text, LLMs reflect and can amplify societal biases present in the data.
