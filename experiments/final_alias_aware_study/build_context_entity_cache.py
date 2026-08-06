from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from geolexvec.entity_provenance import canonical_protocol_json, parse_protocol_json


FORMULA_VERSION = "unfiltered-average-max-context-entity-cosine-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the unfiltered contextual entity feature matrix."
    )
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--query-context", type=Path, required=True)
    parser.add_argument("--evidence-context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.source_cache, allow_pickle=False) as cached:
        payload: dict[str, Any] = {
            key: np.asarray(cached[key]) for key in cached.files
        }
    with np.load(args.query_context, allow_pickle=False) as cached:
        query_qids = [str(value) for value in cached["qids"]]
        query_offsets = np.asarray(cached["offsets"], dtype=np.int32)
        query_vectors = normalize_rows(np.asarray(cached["vectors"], dtype=np.float32))
        query_anchor_split = str(cached["anchor_split"].item())
        query_held_out = bool(
            cached["held_out_occurrences_used_for_optimization"].item()
        )
        query_protocol = parse_protocol_json(cached["training_protocol_json"].item())
        query_checkpoint = str(cached["checkpoint_sha256"].item())
    with np.load(args.evidence_context, allow_pickle=False) as cached:
        evidence_ids = [str(value) for value in cached["ids"]]
        evidence_offsets = np.asarray(cached["offsets"], dtype=np.int32)
        evidence_vectors = normalize_rows(
            np.asarray(cached["vectors"], dtype=np.float32)
        )
        evidence_anchor_split = str(cached["anchor_split"].item())
        evidence_held_out = bool(
            cached["held_out_occurrences_used_for_optimization"].item()
        )
        evidence_protocol = parse_protocol_json(
            cached["training_protocol_json"].item()
        )
        evidence_checkpoint = str(cached["checkpoint_sha256"].item())

    source_qids = [str(value) for value in payload["qids"]]
    source_evidence_ids = [str(value) for value in payload["evidence_ids"]]
    if source_qids != query_qids or source_evidence_ids != evidence_ids:
        raise RuntimeError("context vectors and source cache use different ID order")
    if query_anchor_split != "train_only" or evidence_anchor_split != "train_only":
        raise RuntimeError("adapted context vectors were not produced by train-only anchors")
    if query_held_out or evidence_held_out:
        raise RuntimeError("adapted context vectors report held-out optimization data")
    if canonical_protocol_json(query_protocol) != canonical_protocol_json(evidence_protocol):
        raise RuntimeError("query and evidence vectors use different training protocols")
    if query_checkpoint != evidence_checkpoint or len(query_checkpoint) != 64:
        raise RuntimeError("query and evidence vectors use different checkpoints")

    features = np.zeros((len(source_qids), len(source_evidence_ids)), dtype=np.float32)
    evidence_profiles = [
        evidence_vectors[int(evidence_offsets[i]) : int(evidence_offsets[i + 1])]
        for i in range(len(source_evidence_ids))
    ]
    for query_index in range(len(source_qids)):
        query_profile = query_vectors[
            int(query_offsets[query_index]) : int(query_offsets[query_index + 1])
        ]
        if len(query_profile):
            for evidence_index, evidence_profile in enumerate(evidence_profiles):
                if len(evidence_profile):
                    similarities = np.clip(query_profile @ evidence_profile.T, 0.0, 1.0)
                    features[query_index, evidence_index] = float(
                        similarities.max(axis=1).mean()
                    )
        if (query_index + 1) % 100 == 0 or query_index + 1 == len(source_qids):
            print(f"context features: {query_index + 1}/{len(source_qids)}", flush=True)

    digest = hashlib.sha256()
    digest.update(FORMULA_VERSION.encode("ascii"))
    for path in (args.source_cache, args.query_context, args.evidence_context):
        digest.update(path.read_bytes())
    payload["fingerprint"] = np.asarray(digest.hexdigest())
    payload["formula_version"] = np.asarray(FORMULA_VERSION)
    payload["context_entity_features"] = features
    payload["variant_names"] = np.asarray(["AliasAwareFusion"])
    payload["entity_variant"] = np.asarray("AliasAwareFusion")
    payload["entity_score_formula"] = np.asarray(
        "mean query-entity maximum non-negative cosine over all evidence entities"
    )
    payload["entity_adaptation_anchor_split"] = np.asarray(
        query_protocol["anchor_split"]
    )
    payload["entity_held_out_occurrences_used_for_optimization"] = np.asarray(
        query_protocol["held_out_occurrences_used_for_optimization"]
    )
    payload["entity_training_protocol_json"] = np.asarray(
        canonical_protocol_json(query_protocol)
    )
    payload["entity_checkpoint_sha256"] = np.asarray(query_checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)

    positive = features[features > 0]
    audit = {
        "formula_version": FORMULA_VERSION,
        "entity_variant": "AliasAwareFusion",
        "identity_constraint": "none",
        "entity_adaptation_anchor_split": query_protocol["anchor_split"],
        "held_out_occurrences_used_for_optimization": query_protocol[
            "held_out_occurrences_used_for_optimization"
        ],
        "training_anchor_count": query_protocol["training_anchor_count"],
        "training_anchor_sha256": query_protocol["training_anchor_sha256"],
        "checkpoint_sha256": query_checkpoint,
        "training_protocol": query_protocol,
        "question_count": len(source_qids),
        "evidence_count": len(source_evidence_ids),
        "context_nonzero_rate": float((features > 0).mean()),
        "context_positive_mean": float(positive.mean()) if len(positive) else 0.0,
        "input_sha256": {
            "source_cache": sha256(args.source_cache),
            "query_context": sha256(args.query_context),
            "evidence_context": sha256(args.evidence_context),
        },
        "output_sha256": sha256(args.out),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
