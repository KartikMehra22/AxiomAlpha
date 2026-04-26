# Phase 6B: RAG System — Financial Knowledge Base
**Research Module 9 of 8 | AxiomAlpha Framework**

---

In institutional research, data is often trapped in unstructured text (news, earnings calls, analyst notes). **Retrieval-Augmented Generation (RAG)** provides the bridge between this unstructured "knowledge" and the quantitative "engine."

### The Strategic Value of RAG
1. **Grounded Reasoning**: Unlike pure LLMs that may hallucinate numbers, RAG retrieves specific historical headlines and returns to ground every answer.
2. **Qualitative Recall**: Researchers can ask complex temporal questions like *"What drove volatility for Banks in early 2020?"* and receive a data-backed synthesis.
3. **Auditability**: Every generated insight is linked to specific source documents ($ID_{art}$, $ID_{sum}$), ensuring a clear line of sight from information to signal.

### Architecture: The Vector Intelligence Loop
**Documents** (Enriched Headlines) $\rightarrow$ **Vectors** (Latent Semantic Space) $\rightarrow$ **Retrieval** (Similarity Search) $\rightarrow$ **Synthesis** (Contextual Response).


---
## Document Construction: Enriching the Knowledge Corpus

Raw headlines are too sparse for effective retrieval. To solve this, we implement **Contextual Enrichment**, transforming 13k+ headlines into a structured knowledge graph.

### 1. Document Types
- **Atomic Articles**: Daily headlines enriched with asset returns and market status.
- **Monthly Ticker Summaries**: Aggregated performance and news volume per asset.
- **Sector Quarterly Reports**: High-level themes and risks per industry.

### 2. The Enrichment Formula
Every document is transformed into a rich text block:
$$Doc = \{Date, Asset, Sector, Headline, Sentiment, Ret, Regime\}$$
This ensures that a query about "AAPL volatility" retrieves not just the news, but the price action context surrounding it.


---
## Embedding & Vector Store: Mapping Meaning to Geometry

We map our enriched documents into a high-dimensional **Latent Semantic Space** where documents with similar meanings are mathematically "close."

### 1. TF-IDF Representation
We use the Term Frequency-Inverse Document Frequency weight to represent word importance:
$$W_{t,d} = TF_{t,d} \cdot \log\left(\frac{N}{DF_t}\right)$$

### 2. Dimensionality Reduction (SVD)
To capture semantic relationships rather than just keyword matches, we apply **Singular Value Decomposition (SVD)** to reduce the space to 384 dimensions.
$$A \approx U \Sigma V^T$$

### 3. PCA Projection (Insight)
**Visualization 2** below shows the PCA projection of our 15k documents. We look for **Structural Clustering**: Articles should naturally group by Sector (Tech, Finance) or Sentiment (Positive, Negative) if the embeddings are capturing the underlying financial semantics.


---
## Retrieval Engine: Semantic Search Logic

The retrieval engine identifies the most relevant documents for a given natural language query. We use **Cosine Similarity** to measure the angular distance between the query vector ($q$) and document vectors ($d$).

### Similarity Metric
$$\text{Similarity}(q, d) = \frac{q \cdot d}{\|q\| \|d\|}$$

**Implementation Rationale**:
Due to the environment's specific resource constraints, we utilize a **Vectorized NumPy Search**. This avoids the overhead of external libraries like FAISS while maintaining sub-10ms retrieval latency for our 15k-document corpus.
