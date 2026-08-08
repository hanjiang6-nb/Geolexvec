# Reproducibility

## Fixed Protocol

- Queries: 2,034.
- Evidence candidates: 1,413.
- Source-document groups: 221.
- Outer folds: 5.
- Inner folds per outer training split: 4.
- Entity-vector seeds: 20260725, 20260726, and 20260727.
- Statistical inference: 20,000 source-cluster bootstrap samples and 20,000
  source-cluster sign permutations with Benjamini-Hochberg correction.

## Main Experiment

```bash
python scripts/verify_release.py
python -m pytest -q
python scripts/reproduce_main.py --jobs 3 --iterations 20000
python scripts/reproduce_tables.py --iterations 20000
```

`reproduce_main.py` reruns the fixed nested-validation experiment from the
released feature caches. `reproduce_tables.py` recreates Tables 4 and 5 from
the compact locked predictions, entity-ablation metrics, and external-baseline
rankings.

## Entity Adapter

Install the optional training dependency:

```bash
python -m pip install -r requirements-training.txt
```

Retrain the three rank-8 entity adapters:

```bash
python experiments/entity_alias_metric_learning/train_canonical_alias_encoder.py \
  --vectors data/raw/entity_training/attn_vectors.npy \
  --index data/raw/entity_training/indexed_docs.pkl \
  --aliases data/raw/entity_training/en_zh_equivalence_audit.json \
  --occurrences data/raw/entity_training/context_occurrences.npy \
  --metadata data/raw/entity_training/context_occurrences_metadata.json \
  --out-dir reproduced/entity_adapter \
  --seeds 20260725 20260726 20260727 \
  --rank 8 --epochs 300 --learning-rate 0.002 \
  --temperature 0.07 --margin 0.20 --initial-gate 0.5 \
  --device auto
```

Transform the released query and evidence occurrence vectors with a trained
checkpoint:

```bash
python experiments/entity_alias_metric_learning/transform_canonical_context_vectors.py \
  --checkpoint reproduced/entity_adapter/seed_20260725/canonical_alias_encoder.pt \
  --aliases data/raw/entity_training/en_zh_equivalence_audit.json \
  --training-occurrences data/raw/entity_training/context_occurrences.npy \
  --training-metadata data/raw/entity_training/context_occurrences_metadata.json \
  --query-input data/raw/entity_training/query_context_vectors_epoch200_wordid.npz \
  --evidence-input data/raw/entity_training/evidence_context_vectors_epoch200_wordid.npz \
  --query-output reproduced/entity_context/seed_20260725/query.npz \
  --evidence-output reproduced/entity_context/seed_20260725/evidence.npz \
  --device cpu
```

Each checkpoint and feature cache contains the training protocol and input
hashes required by `scripts/verify_release.py`.
