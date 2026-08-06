from __future__ import annotations

from geolexvec.entity_aliases import CanonicalSurfaceResolver, canonical_group_ids


def test_case_sensitive_formula_is_not_mapped_to_hafnium() -> None:
    payload = {
        "mappings": [
            {"source": "Hf", "target": "铪", "accepted": True},
            {"source": "Li", "target": "锂", "accepted": True},
        ]
    }
    resolver = CanonicalSurfaceResolver(payload)
    assert resolver.resolve("Hf") == "铪"
    assert resolver.resolve("HF") is None
    assert resolver.resolve("li") == "锂"
    assert canonical_group_ids(["Hf", "HF"], payload, ["铪"]).tolist() == [0, -1]
