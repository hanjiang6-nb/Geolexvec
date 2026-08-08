from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
ENTITY_TRAINING_DIR = BASE_DIR / "experiments" / "entity_alias_metric_learning"
if str(ENTITY_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(ENTITY_TRAINING_DIR))

from entity_training_utils import (
    LowRankResidualProjection,
    eligible_pairs,
    load_alias_pairs,
    load_occurrences,
    normalize_rows,
    sha256,
)
from train_canonical_alias_encoder import (
    initialize_prototypes,
)
from geolexvec.entity_aliases import canonical_group_ids
from geolexvec.entity_provenance import canonical_protocol_json


VARIANTS = ("residual_only", "prototype_only")
SEEDS = (20260725, 20260726, 20260727)


class IdentityBase(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(values, dim=-1)


class InternalVariantEncoder(torch.nn.Module):
    def __init__(
        self,
        dimension: int,
        rank: int,
        initial_prototypes: np.ndarray,
        initial_gate: float,
        seed: int,
        variant: str,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unsupported trainable variant: {variant}")
        self.variant = variant
        self.use_residual = variant == "residual_only"
        self.use_prototype = variant == "prototype_only"
        self.base = (
            LowRankResidualProjection(dimension, rank, seed)
            if self.use_residual
            else IdentityBase()
        )
        self.prototypes = torch.nn.Parameter(
            torch.from_numpy(normalize_rows(initial_prototypes)).clone()
        )
        if self.use_prototype:
            initial_gate = min(max(initial_gate, 1e-4), 1.0 - 1e-4)
            self.gate_logit = torch.nn.Parameter(
                torch.tensor(
                    np.log(initial_gate / (1.0 - initial_gate)),
                    dtype=torch.float32,
                )
            )

    @property
    def gate(self) -> torch.Tensor:
        if not self.use_prototype:
            return torch.tensor(0.0, device=self.prototypes.device)
        return torch.sigmoid(self.gate_logit)

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1)

    def base_transform(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(F.normalize(values, dim=-1))

    def encode(self, values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        base = self.base_transform(values)
        if not self.use_prototype:
            return base
        output = base.clone()
        mask = group_ids >= 0
        if bool(mask.any()):
            prototypes = self.normalized_prototypes()[group_ids[mask]]
            output[mask] = F.normalize(
                (1.0 - self.gate) * base[mask] + self.gate * prototypes,
                dim=-1,
            )
        return output


def train_variant(
    grouped: dict[str, dict[str, list[np.ndarray]]],
    pairs: list[Any],
    anchors: np.ndarray,
    variant: str,
    seed: int,
    rank: int,
    epochs: int,
    learning_rate: float,
    temperature: float,
    margin: float,
    device: torch.device,
) -> tuple[InternalVariantEncoder, list[str], dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    group_names = sorted({pair.target for pair in pairs})
    group_index = {name: index for index, name in enumerate(group_names)}
    prototypes = initialize_prototypes(grouped, pairs, group_names)
    model = InternalVariantEncoder(
        anchors.shape[1], rank, prototypes, 0.5, seed, variant
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    group_ids = torch.tensor(
        [group_index[pair.target] for pair in pairs], dtype=torch.long, device=device
    )
    anchor_count = min(1024, len(anchors))
    anchor_indices = random.Random(seed + 41).sample(range(len(anchors)), anchor_count)
    anchor_values = torch.from_numpy(
        normalize_rows(anchors[anchor_indices])
    ).to(device)
    unknown_ids = torch.full(
        (anchor_count,), -1, dtype=torch.long, device=device
    )
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        source_values = torch.from_numpy(
            normalize_rows(
                np.stack([rng.choice(grouped["train"][pair.source]) for pair in pairs])
            )
        ).to(device)
        target_values = torch.from_numpy(
            normalize_rows(
                np.stack([rng.choice(grouped["train"][pair.target]) for pair in pairs])
            )
        ).to(device)
        source_base = model.base_transform(source_values)
        target_base = model.base_transform(target_values)
        source_encoded = model.encode(source_values, group_ids)
        target_encoded = model.encode(target_values, group_ids)
        prototypes_normalized = model.normalized_prototypes()

        source_logits = source_encoded @ prototypes_normalized.T / temperature
        target_logits = target_encoded @ prototypes_normalized.T / temperature
        classification = 0.5 * (
            F.cross_entropy(source_logits, group_ids)
            + F.cross_entropy(target_logits, group_ids)
        )
        alias_similarity = (source_encoded * target_encoded).sum(dim=1)
        alignment = (1.0 - alias_similarity).mean()
        raw_scores = source_encoded @ prototypes_normalized.T
        positive = raw_scores[
            torch.arange(len(group_ids), device=device), group_ids
        ]
        negative_scores = raw_scores.clone()
        negative_scores[
            torch.arange(len(group_ids), device=device), group_ids
        ] = -2.0
        hardest_negative = negative_scores.max(dim=1).values
        ranking = F.relu(hardest_negative - positive + margin).mean()
        anchor_base = model.encode(anchor_values, unknown_ids)
        base_preservation = (1.0 - (anchor_base * anchor_values).sum(dim=1)).mean()
        context_preservation = 0.5 * (
            (1.0 - (source_encoded * source_base).sum(dim=1)).mean()
            + (1.0 - (target_encoded * target_base).sum(dim=1)).mean()
        )
        loss = (
            classification
            + ranking
            + 0.5 * alignment
            + 0.5 * base_preservation
            + 0.2 * context_preservation
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "classification": float(classification.detach()),
                    "ranking": float(ranking.detach()),
                    "alignment": float(alignment.detach()),
                    "base_preservation": float(base_preservation.detach()),
                    "context_preservation": float(context_preservation.detach()),
                    "gate": float(model.gate.detach()),
                }
            )

    report = {
        "variant": variant,
        "seed": seed,
        "learned_gate": float(model.gate.detach()),
    }
    return model, group_names, report


def transform_npz(
    input_path: Path,
    output_path: Path,
    model: InternalVariantEncoder,
    group_names: list[str],
    aliases_path: Path,
    variant: str,
    seed: int,
    device: torch.device,
    training_protocol: dict[str, Any],
    checkpoint_sha256: str,
) -> None:
    alias_payload = json.loads(aliases_path.read_text(encoding="utf-8"))
    with np.load(input_path, allow_pickle=False) as cached:
        payload = {key: np.asarray(cached[key]) for key in cached.files}
    values = normalize_rows(np.asarray(payload["vectors"], dtype=np.float32))
    group_ids = canonical_group_ids(payload["surfaces"], alias_payload, group_names)
    model.eval()
    with torch.no_grad():
        vector_tensor = torch.from_numpy(values).to(device)
        id_tensor = torch.from_numpy(group_ids).to(device)
        output = model.encode(vector_tensor, id_tensor).cpu().numpy()
    payload["vectors"] = output.astype(np.float32)
    payload["adaptation"] = np.asarray(variant)
    payload["gate"] = np.asarray(float(model.gate.detach()))
    payload["seed"] = np.asarray(seed)
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
    parser = argparse.ArgumentParser(description="Train internal entity-adapter ablations")
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--occurrences", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raw-query-dir", type=Path, required=True)
    parser.add_argument("--raw-evidence-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    vectors = np.load(args.vectors, allow_pickle=True).item()
    pairs = load_alias_pairs(args.aliases)
    grouped, _occurrence_audit = load_occurrences(args.occurrences, args.metadata)
    train_pairs, test_pairs = eligible_pairs(pairs, grouped)
    train_anchor_count = sum(len(values) for values in grouped["train"].values())
    held_out_occurrence_count = sum(len(values) for values in grouped["test"].values())
    anchors = np.stack(
        [value for values_for_surface in grouped["train"].values() for value in values_for_surface]
    ).astype(np.float32)
    if len(anchors) != train_anchor_count:
        raise AssertionError("base-preservation anchors must come from the training split only")

    input_hashes = {
        "vectors": sha256(args.vectors),
        "index": sha256(args.index),
        "aliases": sha256(args.aliases),
        "occurrences": sha256(args.occurrences),
        "metadata": sha256(args.metadata),
    }
    anchor_hash = hashlib.sha256(np.ascontiguousarray(anchors).tobytes()).hexdigest()

    for variant in VARIANTS:
        for seed in SEEDS:
            model, group_names, report = train_variant(
                grouped,
                train_pairs,
                anchors,
                variant,
                seed,
                args.rank,
                args.epochs,
                args.learning_rate,
                args.temperature,
                args.margin,
                device,
            )
            variant_seed_dir = args.out_dir / "trained" / variant / f"seed_{seed}"
            variant_seed_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "state_dict": model.state_dict(),
                "dimension": anchors.shape[1],
                "rank": args.rank,
                "seed": seed,
                "group_names": group_names,
                "variant": variant,
                "learned_gate": float(model.gate.detach()),
                "training_protocol": {
                    "schema": "geolexvec.entity-adapter.training.v1",
                    "variant": variant,
                    "anchor_split": "train_only",
                    "training_anchor_count": train_anchor_count,
                    "held_out_occurrence_count": held_out_occurrence_count,
                    "held_out_occurrences_used_for_optimization": False,
                    "training_anchor_sha256": anchor_hash,
                    "input_sha256": input_hashes,
                    "qa_relevance_labels_used": False,
                    "rank": args.rank,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "temperature": args.temperature,
                    "margin": args.margin,
                    "initial_gate": 0.5,
                },
            }
            checkpoint_path = variant_seed_dir / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            checkpoint_hash = sha256(checkpoint_path)
            report["trainable_pairs"] = len(train_pairs)
            report["testable_pairs"] = len(test_pairs)
            report["schema"] = "geolexvec.entity-adapter.internal-report.v1"
            report["protocol"] = checkpoint["training_protocol"]
            report["checkpoint_sha256"] = checkpoint_hash
            (variant_seed_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            transform_npz(
                args.raw_query_dir / f"query_context_vectors_epoch200_wordid.npz",
                args.out_dir / "vectors" / variant / f"seed_{seed}" / "query.npz",
                model,
                group_names,
                args.aliases,
                variant,
                seed,
                device,
                checkpoint["training_protocol"],
                checkpoint_hash,
            )
            transform_npz(
                args.raw_evidence_dir / f"evidence_context_vectors_epoch200_wordid.npz",
                args.out_dir / "vectors" / variant / f"seed_{seed}" / "evidence.npz",
                model,
                group_names,
                args.aliases,
                variant,
                seed,
                device,
                checkpoint["training_protocol"],
                checkpoint_hash,
            )
            print(
                f"completed {variant} seed={seed} gate={float(model.gate.detach()):.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
