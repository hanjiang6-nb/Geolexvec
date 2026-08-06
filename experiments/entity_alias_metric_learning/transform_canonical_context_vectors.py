from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from entity_training_utils import normalize_rows
from train_canonical_alias_encoder import CanonicalAliasEncoder
from geolexvec.entity_aliases import canonical_group_ids
from geolexvec.entity_provenance import (
    canonical_protocol_json,
    validate_training_protocol,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(
    path: Path, device: torch.device
) -> tuple[CanonicalAliasEncoder, list[str], dict[str, Any], str]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    protocol = validate_training_protocol(checkpoint.get("training_protocol"))
    group_names = [str(value) for value in checkpoint["group_names"]]
    dimension = int(checkpoint["dimension"])
    model = CanonicalAliasEncoder(
        dimension=dimension,
        rank=int(checkpoint["rank"]),
        initial_prototypes=np.ones((len(group_names), dimension), dtype=np.float32),
        initial_gate=0.5,
        seed=int(checkpoint["seed"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, group_names, protocol, sha256(path)


def verify_training_inputs(
    protocol: dict[str, Any], aliases: Path, occurrences: Path, metadata: Path
) -> None:
    expected = protocol["input_sha256"]
    actual = {
        "aliases": sha256(aliases),
        "occurrences": sha256(occurrences),
        "metadata": sha256(metadata),
    }
    mismatches = [name for name in actual if actual[name] != expected[name]]
    if mismatches:
        raise ValueError(
            "checkpoint training provenance does not match released inputs: "
            + ", ".join(mismatches)
        )


def transform(
    input_path: Path,
    output_path: Path,
    model: CanonicalAliasEncoder,
    group_names: list[str],
    aliases_path: Path,
    training_protocol: dict[str, Any],
    checkpoint_sha256: str,
    device: torch.device,
    batch_size: int,
) -> None:
    with np.load(input_path, allow_pickle=False) as cached:
        payload = {key: np.asarray(cached[key]) for key in cached.files}
    alias_payload = json.loads(aliases_path.read_text(encoding="utf-8"))
    group_ids = canonical_group_ids(
        payload["surfaces"], alias_payload, group_names
    )
    values = normalize_rows(np.asarray(payload["vectors"], dtype=np.float32))
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            encoded = model.encode(
                torch.from_numpy(values[start:stop]).to(device),
                torch.from_numpy(group_ids[start:stop]).to(device),
            )
            rows.append(encoded.cpu().numpy().astype(np.float32))
    payload["vectors"] = np.concatenate(rows, axis=0)
    payload["adaptation"] = np.asarray("canonical_alias_train_only")
    payload["gate"] = np.asarray(float(model.gate.detach().cpu()))
    payload["anchor_split"] = np.asarray(training_protocol["anchor_split"])
    payload["held_out_occurrences_used_for_optimization"] = np.asarray(
        training_protocol["held_out_occurrences_used_for_optimization"]
    )
    payload["training_protocol_json"] = np.asarray(
        canonical_protocol_json(training_protocol)
    )
    payload["checkpoint_sha256"] = np.asarray(checkpoint_sha256)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a trained canonical alias encoder to context-vector NPZ files."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--training-occurrences", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--query-input", type=Path, required=True)
    parser.add_argument("--evidence-input", type=Path, required=True)
    parser.add_argument("--query-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, group_names, training_protocol, checkpoint_hash = load_model(
        args.checkpoint, device
    )
    verify_training_inputs(
        training_protocol,
        args.aliases,
        args.training_occurrences,
        args.training_metadata,
    )
    transform(
        args.query_input,
        args.query_output,
        model,
        group_names,
        args.aliases,
        training_protocol,
        checkpoint_hash,
        device,
        args.batch_size,
    )
    transform(
        args.evidence_input,
        args.evidence_output,
        model,
        group_names,
        args.aliases,
        training_protocol,
        checkpoint_hash,
        device,
        args.batch_size,
    )
    print(
        f"wrote adapted context vectors with gate={float(model.gate.detach().cpu()):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
