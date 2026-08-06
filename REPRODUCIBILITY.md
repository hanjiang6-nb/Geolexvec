# Reproducibility Checklist

| Requirement | Location | Included |
|---|---|---:|
| All 2,034 queries | `data/raw/benchmark/questions_2034.jsonl` | Yes |
| All 1,413 candidate sentences | `data/raw/benchmark/evidence_1413.jsonl` | Yes |
| Metadata for 356 source articles | `data/raw/benchmark/articles_356.jsonl` | Yes |
| Source-document groups | `data/evaluation_gold.jsonl` and query field `source_doc_id` | Yes |
| Fixed five outer and four inner folds | `data/splits/source_grouped_5x4.json` | Yes |
| 62 accepted aliases and canonical mappings | `data/raw/entity_training/en_zh_equivalence_audit.json` | Yes |
| Geological phrase groups | `data/config/keyword_groups.json` | Yes |
| Five entity-adaptation losses | `experiments/entity_alias_metric_learning/train_canonical_alias_encoder.py` | Yes |
| Checkpoint-to-context-vector transformation | `experiments/entity_alias_metric_learning/transform_canonical_context_vectors.py` | Yes |
| Contextual entity-cache reconstruction | `experiments/final_alias_aware_study/build_context_entity_cache.py` | Yes |
| External baseline checkpoints and parameters | `BASELINES.md` | Yes |
| Locked external baseline rankings | `data/baselines/` | Yes |
| Three-seed out-of-fold predictions | `results/three_module_seeds/seed_*/per_question_metrics.csv` | Yes |
| Cluster bootstrap and sign-permutation code | `geolexvec/validation.py` | Yes |
| One-command Tables 4 and 5 | `scripts/reproduce_tables.py` | Yes |

## Locked Protocol

- Queries: 2,034.
- Evidence candidates: 1,413.
- Source-article records: 356; every candidate has a valid `article_id`.
- Source-document groups: 221.
- Outer folds: 5.
- Inner folds inside each outer training split: 4.
- Entity-vector seeds: 20260725, 20260726, and 20260727.
- Uncertainty: 20,000 source-cluster bootstrap samples and 20,000
  source-cluster sign permutations, followed by Benjamini-Hochberg correction.

## Entity Resource Counts

The public mapping audit contains 62 accepted alias pairs. With the contextual
occurrence export used by the final model, 61 pairs have trainable occurrences.
The base-preservation loss samples only 6,605 training occurrences. None of the
1,600 held-out occurrences is used for parameter optimization; 54 testable
alias relations cover 274 of those held-out contexts for representation-level
evaluation. These counts belong to the final checkpoints in
`data/processed/entity_vectors/`.

The entity adapter uses rank `8`, 300 epochs, learning rate `0.002`, temperature
`0.07`, hard-negative margin `0.20`, and an initial prototype gate of `0.5`.
The gate is learned jointly with the residual transformation and prototypes.
Each generated feature-cache protocol records `entity_adaptation_anchor_split`
as `train_only` and `held_out_occurrences_used_for_optimization` as `false`.

## Commands

```bash
# Verify data, predictions, code, hashes, and sensitive-path hygiene.
python scripts/verify_release.py

# Recompute all three-seed nested-validation predictions.
python scripts/reproduce_main.py --jobs 3 --iterations 20000

# Recreate manuscript Tables 4 and 5 from locked predictions and baseline runs.
python scripts/reproduce_tables.py

# Re-export the fixed fold manifest.
python scripts/build_fixed_splits.py
```

## Rebuild The Entity Adapter

Install the optional training dependency first:

```bash
python -m pip install -r requirements-training.txt
```

Train the three final rank-8 adapters from the released NER occurrence
resources. Only occurrence rows marked `train` are sampled by the optimization
losses; rows marked `test` are used after training for representation-level
evaluation.

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

The released checkpoints and concise evaluation reports are under
`data/processed/entity_vectors/seed_*`. The context-vector transformation and
entity-feature reconstruction entry points are
`transform_canonical_context_vectors.py` and `build_context_entity_cache.py`;
both expose complete path options through `--help`.

For example, transform the seed-20260725 occurrence vectors only after checking
the checkpoint against the released training resources:

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

The transformation fails if the checkpoint lacks the structured train-only
protocol or if any supplied training-resource hash differs from the values
embedded in the checkpoint.
