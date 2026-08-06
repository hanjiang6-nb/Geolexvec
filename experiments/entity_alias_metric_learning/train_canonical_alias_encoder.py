from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from entity_training_utils import (
    AliasPair,
    LowRankResidualProjection,
    eligible_pairs,
    load_alias_pairs,
    load_occurrences,
    normalize_rows,
    sha256,
)


DEFAULT_SEEDS = (20260725, 20260726, 20260727)
PROTOCOL_SCHEMA = "geolexvec.entity-adapter.training.v1"


class CanonicalAliasEncoder(torch.nn.Module):
    def __init__(
        self,
        dimension: int,
        rank: int,
        initial_prototypes: np.ndarray,
        initial_gate: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.base = LowRankResidualProjection(dimension, rank, seed)
        self.prototypes = torch.nn.Parameter(
            torch.from_numpy(normalize_rows(initial_prototypes)).clone()
        )
        initial_gate = min(max(initial_gate, 1e-4), 1.0 - 1e-4)
        self.gate_logit = torch.nn.Parameter(
            torch.tensor(math.log(initial_gate / (1.0 - initial_gate)), dtype=torch.float32)
        )

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def normalized_prototypes(self) -> torch.Tensor:
        return functional.normalize(self.prototypes, dim=-1)

    def encode(self, values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        base = self.base(values)
        output = base.clone()
        mask = group_ids >= 0
        if bool(mask.any()):
            prototypes = self.normalized_prototypes()[group_ids[mask]]
            output[mask] = functional.normalize(
                (1.0 - self.gate) * base[mask] + self.gate * prototypes,
                dim=-1,
            )
        return output


def initialize_prototypes(
    grouped: dict[str, dict[str, list[np.ndarray]]],
    pairs: list[AliasPair],
    group_names: list[str],
) -> np.ndarray:
    rows = []
    sources_by_target: dict[str, set[str]] = {name: {name} for name in group_names}
    for pair in pairs:
        sources_by_target[pair.target].add(pair.source)
    for target in group_names:
        values = [
            value
            for surface in sorted(sources_by_target[target])
            for value in grouped["train"].get(surface, [])
        ]
        rows.append(normalize_rows(np.stack(values)).mean(axis=0))
    return normalize_rows(np.stack(rows))


def train_encoder(
    grouped: dict[str, dict[str, list[np.ndarray]]],
    pairs: list[AliasPair],
    anchors: np.ndarray,
    rank: int,
    epochs: int,
    learning_rate: float,
    temperature: float,
    margin: float,
    base_preservation_weight: float,
    context_preservation_weight: float,
    initial_gate: float,
    seed: int,
    device: torch.device,
) -> tuple[CanonicalAliasEncoder, list[str], list[dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    dimension = anchors.shape[1]
    group_names = sorted({pair.target for pair in pairs})
    group_index = {name: index for index, name in enumerate(group_names)}
    prototypes = initialize_prototypes(grouped, pairs, group_names)
    model = CanonicalAliasEncoder(
        dimension, rank, prototypes, initial_gate, seed
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    group_ids = torch.tensor(
        [group_index[pair.target] for pair in pairs], dtype=torch.long, device=device
    )
    anchor_count = min(1024, len(anchors))
    anchor_indices = random.Random(seed + 41).sample(range(len(anchors)), anchor_count)
    anchor_values = torch.from_numpy(normalize_rows(anchors[anchor_indices])).to(device)
    unknown_ids = torch.full((anchor_count,), -1, dtype=torch.long, device=device)
    history = []

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
        source_base = model.base(source_values)
        target_base = model.base(target_values)
        source_encoded = model.encode(source_values, group_ids)
        target_encoded = model.encode(target_values, group_ids)
        prototypes_normalized = model.normalized_prototypes()
        source_logits = source_encoded @ prototypes_normalized.T / temperature
        target_logits = target_encoded @ prototypes_normalized.T / temperature
        classification = 0.5 * (
            functional.cross_entropy(source_logits, group_ids)
            + functional.cross_entropy(target_logits, group_ids)
        )
        alias_similarity = (source_encoded * target_encoded).sum(dim=1)
        alignment = (1.0 - alias_similarity).mean()
        raw_scores = source_encoded @ prototypes_normalized.T
        positive = raw_scores[torch.arange(len(group_ids), device=device), group_ids]
        negative_scores = raw_scores.clone()
        negative_scores[torch.arange(len(group_ids), device=device), group_ids] = -2.0
        hardest_negative = negative_scores.max(dim=1).values
        ranking = functional.relu(hardest_negative - positive + margin).mean()
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
            + base_preservation_weight * base_preservation
            + context_preservation_weight * context_preservation
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
                    "alias_similarity": float(alias_similarity.mean().detach()),
                    "hardest_negative": float(hardest_negative.mean().detach()),
                    "base_preservation": float(base_preservation.detach()),
                    "context_preservation": float(context_preservation.detach()),
                    "gate": float(model.gate.detach()),
                }
            )
    return model, group_names, history


def encode_numpy(
    model: CanonicalAliasEncoder,
    values: np.ndarray,
    group_ids: np.ndarray,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        tensor = torch.from_numpy(normalize_rows(values)).to(device)
        ids = torch.from_numpy(np.asarray(group_ids, dtype=np.int64)).to(device)
        return model.encode(tensor, ids).cpu().numpy()


def surface_group_map(pairs: list[AliasPair], group_names: list[str]) -> dict[str, int]:
    group_index = {name: index for index, name in enumerate(group_names)}
    result = {name: index for name, index in group_index.items()}
    for pair in pairs:
        result[pair.source] = group_index[pair.target]
        result[pair.target] = group_index[pair.target]
    return result


def evaluate_test(
    model: CanonicalAliasEncoder,
    grouped: dict[str, dict[str, list[np.ndarray]]],
    train_pairs: list[AliasPair],
    test_pairs: list[AliasPair],
    group_names: list[str],
) -> dict[str, Any]:
    mapping = surface_group_map(train_pairs, group_names)
    prototypes = model.normalized_prototypes().detach().cpu().numpy()
    ranks = []
    positives = []
    negatives = []
    details = []
    for pair in test_pairs:
        group_id = mapping[pair.source]
        values = np.stack(grouped["test"][pair.source])
        encoded = encode_numpy(
            model, values, np.full(len(values), group_id, dtype=np.int64)
        )
        scores = encoded @ prototypes.T
        pair_ranks = []
        pair_positives = []
        pair_negatives = []
        predictions = Counter()
        for row in scores:
            order = np.argsort(-row, kind="stable")
            rank_value = int(np.flatnonzero(order == group_id)[0]) + 1
            positive = float(row[group_id])
            hardest_negative = float(np.delete(row, group_id).max())
            ranks.append(rank_value)
            positives.append(positive)
            negatives.append(hardest_negative)
            pair_ranks.append(rank_value)
            pair_positives.append(positive)
            pair_negatives.append(hardest_negative)
            predictions[group_names[int(order[0])]] += 1
        details.append(
            {
                "source": pair.source,
                "target": pair.target,
                "test_occurrences": len(pair_ranks),
                "recall_at_1": float(np.mean(np.asarray(pair_ranks) == 1)),
                "mrr": float(np.mean(1.0 / np.asarray(pair_ranks))),
                "mean_positive_similarity": float(np.mean(pair_positives)),
                "mean_hardest_negative_similarity": float(np.mean(pair_negatives)),
                "mean_margin": float(np.mean(np.asarray(pair_positives) - np.asarray(pair_negatives))),
                "top_predictions": predictions.most_common(5),
            }
        )
    rank_array = np.asarray(ranks)
    positive_array = np.asarray(positives)
    negative_array = np.asarray(negatives)
    return {
        "test_occurrences": len(ranks),
        "test_alias_pairs": len(test_pairs),
        "recall_at_1": float(np.mean(rank_array == 1)),
        "recall_at_5": float(np.mean(rank_array <= 5)),
        "mrr": float(np.mean(1.0 / rank_array)),
        "mean_positive_similarity": float(positive_array.mean()),
        "mean_hardest_negative_similarity": float(negative_array.mean()),
        "mean_margin": float(np.mean(positive_array - negative_array)),
        "details": details,
    }


def transform_outputs(
    model: CanonicalAliasEncoder,
    group_names: list[str],
    train_pairs: list[AliasPair],
    vectors: dict[str, Any],
    index_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    mapping = surface_group_map(train_pairs, group_names)
    names = sorted(vectors)
    values = np.stack([np.asarray(vectors[name], dtype=np.float32) for name in names])
    group_ids = np.asarray([mapping.get(name, -1) for name in names], dtype=np.int64)
    encoded = encode_numpy(model, values, group_ids)
    output_vectors = {name: encoded[index] for index, name in enumerate(names)}
    vector_out = output_dir / "attn_vectors_canonical_alias_learned.npy"
    np.save(vector_out, output_vectors, allow_pickle=True)

    with index_path.open("rb") as handle:
        indexed_docs = pickle.load(handle)
    occurrence_count = 0
    for document in indexed_docs:
        entities = document.get("entity_vectors", [])
        valid = []
        raw_values = []
        ids = []
        for index, entity in enumerate(entities):
            value = np.asarray(entity.get("vector"), dtype=np.float32).reshape(-1)
            if value.size != encoded.shape[1]:
                continue
            surface = str(entity.get("text") or "").strip()
            valid.append(index)
            raw_values.append(value)
            ids.append(mapping.get(surface, -1))
        if raw_values:
            transformed = encode_numpy(
                model, np.stack(raw_values), np.asarray(ids, dtype=np.int64)
            )
            for index, value in zip(valid, transformed):
                entities[index]["vector"] = value
                occurrence_count += 1
    index_out = output_dir / "indexed_docs_canonical_alias_learned.pkl"
    with index_out.open("wb") as handle:
        pickle.dump(indexed_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    text_out = output_dir / "attn_vectors_canonical_alias_learned_256d.txt"
    with text_out.open("w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write(f"# vector_count: {len(output_vectors)}\n")
        handle.write(f"# learned_alias_gate: {float(model.gate.detach()):.9f}\n")
        handle.write("# columns: entity_text\\tvector\n")
        for name in names:
            serialized = " ".join(f"{float(value):.8g}" for value in output_vectors[name])
            handle.write(f"{name}\t{serialized}\n")
    return {
        "vectors": vector_out,
        "index": index_out,
        "text": text_out,
        "occurrence_count": occurrence_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised canonical alias encoder.")
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--occurrences", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--base-preservation-weight", type=float, default=0.50)
    parser.add_argument("--context-preservation-weight", type=float, default=0.20)
    parser.add_argument("--initial-gate", type=float, default=0.50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    paths = {
        "vectors": args.vectors.resolve(),
        "index": args.index.resolve(),
        "aliases": args.aliases.resolve(),
        "occurrences": args.occurrences.resolve(),
        "metadata": args.metadata.resolve(),
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    vectors = np.load(paths["vectors"], allow_pickle=True).item()
    pairs = load_alias_pairs(paths["aliases"])
    grouped, occurrence_audit = load_occurrences(paths["occurrences"], paths["metadata"])
    train_pairs, test_pairs = eligible_pairs(pairs, grouped)
    train_anchor_count = sum(len(values) for values in grouped["train"].values())
    held_out_occurrence_count = sum(len(values) for values in grouped["test"].values())
    anchors = np.stack(
        [value for values in grouped["train"].values() for value in values]
    ).astype(np.float32)
    if len(anchors) != train_anchor_count:
        raise AssertionError("base-preservation anchors must come from the training split only")
    training_protocol = {
        "schema": PROTOCOL_SCHEMA,
        "anchor_split": "train_only",
        "training_anchor_count": train_anchor_count,
        "held_out_occurrence_count": held_out_occurrence_count,
        "held_out_occurrences_used_for_optimization": False,
        "training_anchor_sha256": hashlib.sha256(
            np.ascontiguousarray(anchors).tobytes()
        ).hexdigest(),
        "input_sha256": hashes_before,
        "qa_relevance_labels_used": False,
        "rank": args.rank,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "margin": args.margin,
        "base_preservation_weight": args.base_preservation_weight,
        "context_preservation_weight": args.context_preservation_weight,
        "initial_gate": args.initial_gate,
    }
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    out_dir = args.out_dir.resolve()
    seed_reports = []
    print(
        f"device={device} train_pairs={len(train_pairs)} test_pairs={len(test_pairs)} "
        f"training_anchors={len(anchors)} held_out_occurrences={held_out_occurrence_count}",
        flush=True,
    )

    for seed in args.seeds:
        model, group_names, history = train_encoder(
            grouped,
            train_pairs,
            anchors,
            args.rank,
            args.epochs,
            args.learning_rate,
            args.temperature,
            args.margin,
            args.base_preservation_weight,
            args.context_preservation_weight,
            args.initial_gate,
            seed,
            device,
        )
        evaluation = evaluate_test(model, grouped, train_pairs, test_pairs, group_names)
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = seed_dir / "canonical_alias_encoder.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "dimension": anchors.shape[1],
                "rank": args.rank,
                "seed": seed,
                "group_names": group_names,
                "learned_gate": float(model.gate.detach()),
                "training_protocol": training_protocol,
            },
            checkpoint_path,
        )
        outputs = transform_outputs(
            model, group_names, train_pairs, vectors, paths["index"], seed_dir
        )
        report = {
            "schema": "geolexvec.entity-adapter.report.v1",
            "seed": seed,
            "learned_gate": float(model.gate.detach()),
            "group_names": group_names,
            "protocol": training_protocol,
            "representation_evaluation": evaluation,
            "training_history": history,
            "artifacts": {
                "checkpoint": checkpoint_path.name,
                **{
                    key: path.name
                    for key, path in outputs.items()
                    if key != "occurrence_count"
                },
                "transformed_occurrence_count": outputs["occurrence_count"],
            },
            "artifact_sha256": {
                checkpoint_path.name: sha256(checkpoint_path),
                **{
                    path.name: sha256(path)
                for key, path in outputs.items()
                if key != "occurrence_count"
                },
            },
        }
        (seed_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        seed_reports.append(report)
        print(
            f"seed={seed} gate={float(model.gate.detach()):.4f} "
            f"r1={evaluation['recall_at_1']:.4f} r5={evaluation['recall_at_5']:.4f} "
            f"mrr={evaluation['mrr']:.4f} positive={evaluation['mean_positive_similarity']:.4f} "
            f"negative={evaluation['mean_hardest_negative_similarity']:.4f}",
            flush=True,
        )

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    if hashes_before != hashes_after:
        raise AssertionError("a protected input changed during canonical alias training")
    summary = {
        "schema": "geolexvec.entity-adapter.summary.v1",
        "training_protocol": training_protocol,
        "interpretation": "supervised entity normalization; no claim of discovering unregistered aliases",
        "input_alias_pairs": len(pairs),
        "trainable_alias_pairs": len(train_pairs),
        "testable_alias_pairs": len(test_pairs),
        "occurrence_audit": occurrence_audit,
        "hyperparameters": vars(args) | {"device": str(device)},
        "results": [
            {
                "seed": report["seed"],
                "learned_gate": report["learned_gate"],
                **{
                    key: value
                    for key, value in report["representation_evaluation"].items()
                    if key != "details"
                },
            }
            for report in seed_reports
        ],
        "protected_sha256_before": hashes_before,
        "protected_sha256_after": hashes_after,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote canonical alias summary: {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
