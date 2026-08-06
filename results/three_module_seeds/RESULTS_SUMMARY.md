# GeoLexVec Three-Module Results

The entity adapters were trained from the released NER occurrence resources.
Each checkpoint embeds a structured training protocol containing input hashes,
the 6,605-row training-anchor fingerprint, and an explicit statement that the
1,600 held-out occurrence contexts were not used for optimization. Context
vector transformation validates this protocol and the corresponding training
resources before producing retrieval features.

| Model | Hit@1 | Hit@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Alias-Aware Fusion | 0.774336 | 0.951327 | 0.835400 | 0.666313 |
| Without text | 0.546706 | 0.838414 | 0.644550 | 0.529589 |
| Without entity | 0.676008 | 0.881514 | 0.744136 | 0.604047 |
| Without phrase | 0.743363 | 0.940184 | 0.812230 | 0.651684 |

The mean nested-validation weights remain 0.5827 for text, 0.3373 for entity,
and 0.0800 for phrase. The case-sensitive alias resolver leaves `HF`
(hydrofluoric acid) unregistered instead of mapping it to `Hf` (hafnium).
