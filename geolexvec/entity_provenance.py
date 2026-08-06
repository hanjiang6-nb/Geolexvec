from __future__ import annotations

import json
from typing import Any


PROTOCOL_SCHEMA = "geolexvec.entity-adapter.training.v1"


def validate_training_protocol(protocol: Any) -> dict[str, Any]:
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("missing structured entity-adapter training protocol")
    if protocol.get("anchor_split") != "train_only":
        raise ValueError("entity adapter was not trained with train-only anchors")
    if protocol.get("held_out_occurrences_used_for_optimization") is not False:
        raise ValueError("entity adapter reports held-out optimization data")
    if int(protocol.get("training_anchor_count", 0)) <= 0:
        raise ValueError("entity adapter has no valid training-anchor count")
    fingerprint = str(protocol.get("training_anchor_sha256", ""))
    if len(fingerprint) != 64:
        raise ValueError("entity adapter has no valid training-anchor fingerprint")
    input_hashes = protocol.get("input_sha256")
    if not isinstance(input_hashes, dict):
        raise ValueError("entity adapter has no training-input hashes")
    for name in ("aliases", "occurrences", "metadata"):
        if len(str(input_hashes.get(name, ""))) != 64:
            raise ValueError(f"entity adapter has no valid {name} input hash")
    return protocol


def canonical_protocol_json(protocol: dict[str, Any]) -> str:
    validate_training_protocol(protocol)
    return json.dumps(protocol, sort_keys=True, separators=(",", ":"))


def parse_protocol_json(value: Any) -> dict[str, Any]:
    try:
        protocol = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid entity training protocol JSON") from exc
    return validate_training_protocol(protocol)
