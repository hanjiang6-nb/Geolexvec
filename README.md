# GeoLexVec

GeoLexVec is a three-signal retrieval model for sentence-level coal-geology
evidence retrieval:

```text
score(q, e) = w_text * S_text(q, e)
            + w_entity * S_entity(q, e)
            + w_phrase * S_phrase(q, e)
```

- `S_text`: BAAI/bge-small-zh-v1.5 sentence-vector cosine similarity.
- `S_entity`: average query-entity MaxSim over adapted contextual vectors.
- `S_phrase`: geological phrase-group coverage.

The non-negative signal weights are selected inside source-document-grouped
nested validation. They are not manually assigned from test results.

## Repository Contents

- `data/raw/benchmark/`: 2,034 queries, 1,413 candidate sentences, and 356
  source-link records.
- `data/raw/entity_training/`: inputs required to retrain the entity adapter.
- `data/processed/entity_vectors/`: final three-seed checkpoints and internal
  ablation checkpoints.
- `data/feature_caches/`: final feature matrices used by the main experiment.
- `data/splits/`: fixed five-outer-fold and four-inner-fold partitions.
- `data/baselines/`: locked external-baseline rankings.
- `results/three_module_seeds/`: compact final out-of-fold metrics and selected
  weights for the three entity-vector seeds.
- `results/entity_internal_ablation/`: compact final entity-ablation metrics.
- `results/manuscript_tables/`: final Tables 4 and 5.
- `experiments/`, `geolexvec/`, and `rag_eval/`: model and evaluation code.
- `scripts/`: verification and one-command reproduction entry points.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Reproduction

Verify the released files:

```bash
python scripts/verify_release.py
python -m pytest -q
```

Rerun the three-seed nested-validation experiment:

```bash
python scripts/reproduce_main.py --jobs 3 --iterations 20000
```

Recreate manuscript Tables 4 and 5 from the compact locked predictions and
baseline rankings:

```bash
python scripts/reproduce_tables.py --iterations 20000
```

Detailed entity-adapter retraining commands are provided in
`REPRODUCIBILITY.md`. External model checkpoints and inference parameters are
listed in `BASELINES.md`.

## Expected Main Results

| Model | Strict-Hit@1 | Strict-Hit@10 | Strict-MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| GeoLexVec | 0.774336 | 0.951327 | 0.835400 | 0.666313 |
| w/o text | 0.546706 | 0.838414 | 0.644550 | 0.529589 |
| w/o entity | 0.676008 | 0.881514 | 0.744136 | 0.604047 |
| w/o phrase | 0.743363 | 0.940184 | 0.812230 | 0.651684 |

The strongest locked external baseline, BGE-M3 late interaction, obtains a
Strict-MRR@10 of `0.803482`.

## Licenses

- Software and source code: MIT License.
- GeoLexVec benchmark queries, annotations, and derived metadata: CC BY 4.0.
- Candidate evidence sentences: CoalGeoNER Version 1, CC BY 4.0,
  https://doi.org/10.6084/m9.figshare.32305830.v1.

Complete scientific articles and article abstracts are not redistributed. See
`DATA_LICENSE.md` and `DATA_NOTICE.md`.
