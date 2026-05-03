# NB-09: RAG Setup

## Purpose
Builds a real FAISS retrieval index over synthetic news headlines using `sentence-transformers` (all-MiniLM-L6-v2, 384-dim). Encodes each headline into a normalized 384-dimensional embedding and stores the result in an IndexFlatIP structure where inner product equals cosine similarity for unit vectors.

## Inputs
| Artifact | Producer | Loaded via |
|---|---|---|
| `synthetic_news` | NB08 | `src.io.loaders` |

## Outputs
| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| `embeddings.npy` | `data/processed/embeddings.npy` | N×384 float32 array | NB13, NB15 |
| `faiss_index.bin` | `data/processed/faiss_index.bin` | FAISS IndexFlatIP | NB13, NB15 |
| `rag_documents.json` | `data/processed/rag_documents.json` | list of headline dicts | NB13, NB15 |
| `rag_models.pkl` | `data/processed/rag_models.pkl` | serialized retriever state | NB13, NB15 |

## Key methods
- `SentenceTransformer.encode()`: encodes headlines with `normalize_embeddings=True` producing unit vectors
- `faiss.IndexFlatIP`: inner-product index; cosine similarity for normalized vectors
- `faiss.write_index`: persists index to `faiss_index.bin`

## How to run
```bash
python run_all.py
```

## Verified after fix
- ✅ No `np.random.*` data fabrication
- ✅ Reads exclusively via `src.io.loaders`
- ✅ Outputs schema-validated by downstream loaders
