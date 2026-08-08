# GeoLexVec Main Experiment

This repository contains the reproducible materials for the final
**Alias-Aware Fusion** GeoLexVec experiment on coal-geology evidence retrieval.
The retrieval model has three signals:

```text
score(q, e) = w_text * S_text(q, e)
            + w_entity * S_entity(q, e)
            + w_phrase * S_phrase(q, e)
```

The entity signal is the average, over query entities, of the maximum
non-negative cosine similarity to all evidence entities. Canonical aliases are
used during entity-vector adaptation, but retrieval does not use a hard alias
score, same-canonical-entity filter, or entity-type filter.

## Included Materials

- `data/raw/benchmark/questions_2034.jsonl`: all 2,034 reviewed queries,
  including their source-document links and reference evidence.
- `data/raw/benchmark/evidence_1413.jsonl`: all 1,413 candidate evidence
  sentences used by the benchmark. Every evidence record is linked to a
  concrete source article through `article_id`.
- `data/raw/benchmark/articles_356.jsonl`: title, authors, journal, year,
  keywords, abstract, and other source metadata for the 356 article records.
- `data/evaluation_gold.jsonl`: the locked evaluation labels and source groups.
- `data/splits/source_grouped_5x4.json`: the fixed source-document-grouped
  five-fold outer and four-fold inner partitions.
- `data/raw/entity_training/`: the 62 accepted alias mappings, occurrence
  vectors, alignment metadata, and NER-derived entity resources.
- `data/processed/entity_vectors/`: the three final checkpoints, structured
  train-only provenance reports, and the internal-ablation checkpoints.
- `data/config/keyword_groups.json`: coal-geology phrase groups.
- `data/feature_caches/seed_*_sentence_variants.npz`: the three final
  entity-vector seed caches and their audit protocols.
- `results/three_module_seeds/`: per-question out-of-fold predictions,
  selected fold weights, and significance outputs for all three seeds.
- `results/entity_internal_ablation/`: the raw, residual-only,
  prototype-only, and full-adaptation entity ablation results.

## Reproduction

Install the core dependencies and run the complete three-seed main experiment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/reproduce_main.py --jobs 3
```

The script reruns the fixed five-by-four source-grouped nested validation,
selects weights inside the training folds, evaluates Top-1 through Top-10,
and performs source-cluster bootstrap and sign-permutation tests.

To rebuild Tables 4 and 5 from the locked predictions and baseline runs:

```bash
python scripts/reproduce_tables.py
```

The output is written to `reproduced_tables/`.

The current three-seed means are:

| Model | Strict-Hit@1 | Strict-Hit@10 | Strict-MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Alias-Aware Fusion | 0.774336 | 0.951327 | **0.835400** | 0.666313 |
| w/o text | 0.546706 | 0.838414 | 0.644550 | 0.529589 |
| w/o entity | 0.676008 | 0.881514 | 0.744136 | 0.604047 |
| w/o phrase | 0.743363 | 0.940184 | 0.812230 | 0.651684 |

The nested validation selected mean full-model weights of `0.5827` for text,
`0.3373` for entity, and `0.0800` for phrase. These values are learned from
training folds and are not manually assigned.

## Entity Adaptation Training

The final implementation of the rank-8 residual transformation, canonical
prototype fusion, and all five training losses is
`experiments/entity_alias_metric_learning/train_canonical_alias_encoder.py`.
Shared data and model utilities are in `entity_training_utils.py`. The internal
(2 x 2) ablation runner is
`experiments/final_alias_aware_study/run_entity_internal_ablation.py`.

The accepted alias resource contains 62 pairs. In the final contextual export,
61 are trainable. Base-space preservation samples only the 6,605 training
occurrences; none of the 1,600 held-out occurrences participates in parameter
optimization. Among the held-out occurrences, 274 contexts cover 54 testable
alias relations and are used for representation-level evaluation. The retrieval
benchmark labels are not used to train entity representations.

Each released checkpoint embeds the hashes of its alias, occurrence, metadata,
base-vector, and entity-index inputs, together with the fingerprint of the
6,605 training anchors. Context transformation verifies the checkpoint protocol
and the released training inputs before propagating provenance into the feature
cache.

## External Baselines

External models are not vendored into this repository. Their exact checkpoints,
candidate protocols, and inference parameters are documented in `BASELINES.md`.
The locked baseline rankings are released in `data/baselines/`, allowing Table
4 to be recreated without downloading third-party checkpoints. Cross-Encoder
and BGE-M3 late interaction use BM25 Top-100 candidates; learned sparse BGE-M3
searches all 1,413 candidates.

## Integrity and Scope

The 98 evidence records that originally lacked article links have been
restored by exact normalized sentence matching against article abstracts. The
evidence IDs were updated consistently across the corpus, feature caches,
baseline runs, and released predictions. Candidate order, sentence text,
feature values, scores, and reported metrics are unchanged. Every row in
`evidence_1413.jsonl` retains the same four-field schema: `id`, `source_id`,
`article_id`, and `text`.


Run `python scripts/verify_release.py` and `pytest` to check the package.
`manifest/SHA256SUMS.txt` records file hashes. No server credentials or server
paths are included.

## Licenses

- Software and source code: MIT License.
- GeoLexVec benchmark queries, annotations, and derived metadata: CC BY 4.0.
- Candidate evidence sentences: CoalGeoNER Version 1, CC BY 4.0,
  https://doi.org/10.6084/m9.figshare.32305830.v1.
- Original scientific publications: copyright remains with their respective
  rightsholders; complete articles and article abstracts are not redistributed.

See `DATA_LICENSE.md` and `DATA_NOTICE.md` for the data provenance and
attribution requirements.
