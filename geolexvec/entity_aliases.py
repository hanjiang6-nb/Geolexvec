from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _is_uppercase_two_letter_token(value: str) -> bool:
    """Return true for tokens such as HF whose case can change chemical meaning."""
    return len(value) == 2 and value.isascii() and value.isalpha() and value.isupper()


class CanonicalSurfaceResolver:
    """Resolve registered aliases without collapsing case-sensitive chemical tokens."""

    def __init__(self, alias_payload: dict[str, Any]) -> None:
        self.exact: dict[str, str] = {}
        folded_targets: dict[str, set[str]] = {}
        for row in alias_payload.get("mappings", []):
            if not row.get("accepted"):
                continue
            source = str(row["source"]).strip()
            target = str(row["target"]).strip()
            for surface in (source, target):
                self.exact[surface] = target
                folded_targets.setdefault(surface.casefold(), set()).add(target)
        self.unambiguous_folded = {
            folded: next(iter(targets))
            for folded, targets in folded_targets.items()
            if len(targets) == 1
        }

    def resolve(self, surface: Any) -> str | None:
        value = str(surface).strip()
        if not value:
            return None
        if value in self.exact:
            return self.exact[value]
        # Do not turn HF (hydrofluoric acid) into Hf (hafnium), or make the
        # analogous error for another all-uppercase two-letter formula.
        if _is_uppercase_two_letter_token(value):
            return None
        return self.unambiguous_folded.get(value.casefold())


def canonical_group_ids(
    surfaces: Sequence[Any],
    alias_payload: dict[str, Any],
    group_names: Sequence[str],
) -> np.ndarray:
    resolver = CanonicalSurfaceResolver(alias_payload)
    group_index = {str(name): index for index, name in enumerate(group_names)}
    return np.asarray(
        [group_index.get(resolver.resolve(surface), -1) for surface in surfaces],
        dtype=np.int64,
    )
