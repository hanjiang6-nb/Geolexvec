# Data Notice

This release includes the reviewed benchmark records and the 1,413 candidate
evidence sentences needed to reproduce the reported retrieval experiment.
It also contains standardized metadata for 356 source-article records. All
1,413 evidence sentences are linked to one of these article records; no
placeholder article identifiers remain.
It also includes the 62 accepted alias pairs and the NER-derived occurrence
resources used by entity-vector adaptation. The original source publication
files and server credentials are not included.

The article-link correction changes identifiers only. Sentence text, candidate
order, feature arrays, relevance labels, model scores, and evaluation metrics
are unchanged. Identifier references in the released caches, baseline runs,
and prediction files were updated together.

The two occurrence-level entity indexes are also aligned to the standardized
evidence IDs. Their 1,413-row order, sentence text, vectors, and record schema
are unchanged; 98 placeholder article IDs and the remaining legacy sentence-ID
forms were replaced by the corresponding benchmark `id` and `article_id`.

The final released entity checkpoints were then retrained from the included
train-only occurrence resources with structured input provenance. Correcting
the case-sensitive `HF`/`Hf` mapping changes a small number of entity-feature
values; the bundled predictions and reported metrics were regenerated rather
than reusing the previous run.

The alias resource is used to supervise representation learning. The final
retrieval score compares every query entity with every evidence entity using
the learned vectors; it does not apply a hard alias or same-canonical filter.

The 1,413 candidate evidence sentences are identical, in both content and
order, to the sentence records released in CoalGeoNER Version 1 under CC BY
4.0. This repository does not redistribute complete scientific articles or
article abstracts. The MIT License applies only to software; benchmark data
are governed by `DATA_LICENSE.md`.
