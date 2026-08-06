from __future__ import annotations

import hashlib
from pathlib import Path


EXCLUDED = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".pytest_tmp",
    "reproduced",
    "reproduced_tables",
    "ci_reproduced_tables",
}
TEXT_EXTENSIONS = {
    ".csv", ".json", ".jsonl", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.suffix.lower() in TEXT_EXTENSIONS or path.name in {".gitattributes", ".gitignore"}:
        digest.update(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        return digest.hexdigest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "manifest/SHA256SUMS.txt"
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path != output
        and not EXCLUDED.intersection(path.relative_to(root).parts)
        and path.suffix not in {".pyc", ".pyo"}
    ]
    with output.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(
            "\n".join(
                f"{sha256(path)}  {path.relative_to(root).as_posix()}"
                for path in sorted(files)
            )
            + "\n"
        )
    print(f"wrote {len(files)} checksums")


if __name__ == "__main__":
    main()
