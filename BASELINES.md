# External Baseline Configurations

The released rankings are stored under `data/baselines/`. They allow Table 4
to be rebuilt without downloading external checkpoints. External model code
and checkpoints are not redistributed; this file records the exact implemented
scoring mechanisms and parameters needed for an independent rerun.

| Model | Checkpoint or method | Candidate protocol | Main parameters |
|---|---|---|---|
| BM25 | In-repository implementation | Full 1,413-sentence corpus | `k1=1.5`, `b=0.75`; mixed ASCII token, Chinese character, and Chinese bigram tokenization |
| TF-IDF | scikit-learn | Full corpus | Character 2-4 grams; 20,000 features; sublinear TF |
| Hybrid | BGE-base and BM25 score fusion | Full corpus | Query-level min-max normalization; `0.5/0.5` fusion |
| BGE-base | `BAAI/bge-base-zh-v1.5` | Full corpus | L2-normalized cosine similarity |
| BGE-base + BM25 RRF | BGE-base and BM25 | Full corpus | Reciprocal-rank-fusion constant `60` |
| Cross-Encoder | `BAAI/bge-reranker-v2-m3` | BM25 Top-100 | FP16; batch size `32`; normalized pair score |
| BGE-M3 learned sparse | `BAAI/bge-m3` | Full corpus | FP16; encoding batch size `8`; learned lexical weights |
| BGE-M3 late interaction | `BAAI/bge-m3` | BM25 Top-100 | FP16; encoding batch size `8`; token MaxSim |

The BGE-M3 variants are named by their implemented scoring mechanisms. They
are not presented as independent ColBERT or SPLADE checkpoints.
