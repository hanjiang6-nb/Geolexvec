from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
    "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",
}


@dataclass(frozen=True)
class AliasPair:
    source: str
    target: str
    category: str


class LowRankResidualProjection(torch.nn.Module):
    def __init__(self, dimension: int, rank: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.down = torch.nn.Parameter(
            torch.randn(rank, dimension, generator=generator) / np.sqrt(dimension)
        )
        self.up = torch.nn.Parameter(torch.zeros(dimension, rank))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        delta = (values @ self.down.T) @ self.up.T
        return functional.normalize(values + delta, dim=-1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def load_alias_pairs(path: Path) -> list[AliasPair]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs: list[AliasPair] = []
    for row in payload.get("mappings", []):
        if not row.get("accepted"):
            continue
        source = str(row["source"])
        target = str(row["target"])
        category = (
            "element"
            if source in ELEMENT_SYMBOLS or source in {"A l", "N i", "fe"}
            else "abbreviation"
        )
        pairs.append(AliasPair(source, target, category))
    if not pairs:
        raise ValueError("no accepted alias pairs were loaded")
    return pairs


def load_occurrences(
    matrices_path: Path,
    metadata_path: Path,
) -> tuple[dict[str, dict[str, list[np.ndarray]]], dict[str, Any]]:
    matrices = np.load(matrices_path, allow_pickle=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata["exported"]) != len(matrices):
        raise ValueError("occurrence matrices and alignment metadata have different lengths")
    grouped: dict[str, dict[str, list[np.ndarray]]] = {
        "train": defaultdict(list),
        "test": defaultdict(list),
    }
    status: Counter[str] = Counter()
    for row in metadata["rows"]:
        if not row["exported"]:
            status["skipped"] += 1
            continue
        if not row["aligned"]:
            status["shifted"] += 1
            continue
        split = str(row["split"])
        if split not in grouped:
            raise ValueError(f"unexpected occurrence split: {split}")
        matrix = np.asarray(matrices[int(row["matrix_index"])], dtype=np.float32)
        vector = matrix.mean(axis=0) if matrix.ndim == 2 else matrix.reshape(-1)
        grouped[split][str(row["text"])].append(vector)
        status["aligned"] += 1
    return grouped, {"status": dict(status), "metadata_exported": int(metadata["exported"])}


def eligible_pairs(
    pairs: list[AliasPair],
    grouped: dict[str, dict[str, list[np.ndarray]]],
) -> tuple[list[AliasPair], list[AliasPair]]:
    train_pairs = [
        pair
        for pair in pairs
        if grouped["train"].get(pair.source) and grouped["train"].get(pair.target)
    ]
    test_pairs = [pair for pair in train_pairs if grouped["test"].get(pair.source)]
    if not train_pairs or not test_pairs:
        raise ValueError("alias mappings produced no train/test occurrence pairs")
    return train_pairs, test_pairs
