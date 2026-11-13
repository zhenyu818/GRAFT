#!/usr/bin/env python3
"""Audit the explicit publication allowlist and create a file integrity manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

ROOT_FILES = (
    "README.md",
    "LICENSE",
    "Makefile",
    "go.mod",
    "go.sum",
    "pyproject.toml",
    ".gitignore",
    ".dockerignore",
)
PATTERNS = (
    "api/**/*.go",
    "cmd/**/*.go",
    "internal/**/*.go",
    "python/**/*.py",
    "scripts/*.py",
    "scripts/*.sh",
    "tests/*.py",
    "config/**/*.yaml",
    "build/*.Dockerfile",
    "docs/*.md",
    "results/**/*.json",
    "results/**/*.md",
)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{50,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"/(?:root|home)/[A-Za-z0-9_.-]+/"),
)


def inspect(path, forbidden=()):
    data = path.read_bytes()
    text = data.decode("utf-8")
    if re.search(r"[\u4e00-\u9fff]", text):
        raise ValueError(f"English-language publication check: {path.name}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(data):
            raise ValueError(f"Sensitive content check: {path.name}")
    for token in forbidden:
        if token.encode() in data or token in path.name:
            raise ValueError(f"Publication sentinel detected in {path.suffix} payload")


def positive_controls():
    sentinel = "GRAFT_" + "PUBLICATION_SCAN_CONTROL"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for suffix in (".json", ".md", ".go", ".py", ".yaml", ".Dockerfile"):
            path = root / ("control" + suffix)
            path.write_text(json.dumps({"credential": sentinel}))
            try:
                inspect(path, (sentinel,))
            except ValueError:
                pass
            else:
                raise AssertionError(f"Scanner control must be detected: {suffix}")
        named = root / (sentinel + ".md")
        named.write_text("Control filename")
        try:
            inspect(named, (sentinel,))
        except ValueError:
            pass
        else:
            raise AssertionError("Filename control must be detected")


def files(root):
    selected = {root / name for name in ROOT_FILES}
    for pattern in PATTERNS:
        selected.update(path for path in root.glob(pattern) if "__pycache__" not in str(path))
    return sorted(path for path in selected if path.is_file() and path.name != "release-manifest.json")


def verify(root):
    positive_controls()
    selected = files(root)
    for path in selected:
        inspect(path)
    manifest = {
        "format_version": 1,
        "publication": "GRAFT source and recorded experiments",
        "entries": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in selected
        ],
        "positive_controls": {"content_formats": 6, "filename": True},
    }
    (root / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"audited_files": len(selected), "positive_controls": "passed"}))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    verify(args.root.resolve())
