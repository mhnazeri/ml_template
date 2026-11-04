#!/usr/bin/env python3
"""Rename the template package `project_name` to your project's name, in place.

    python scripts/rename_project.py my_robot_policy
    python scripts/rename_project.py my_robot_policy --description "VLA policy for X" --dry-run

Renames src/project_name/ and rewrites every reference (imports, configs, pyproject,
mise tasks, docs, tests). Only uses the standard library so it runs before `uv sync`.
"""

from __future__ import annotations

import argparse
import keyword
import re
import shutil
import sys
from pathlib import Path

OLD = "project_name"
OLD_CLI_PREFIX = "pn-"
OLD_DESCRIPTION = "Project name, a one sentence description."
ROOT = Path(__file__).resolve().parents[1]
# top-level dirs that are never source (note: src/<pkg>/data/ must NOT be skipped)
SKIP_TOP = {".git", ".venv", "runs", "data", "logs", "build", "dist"}
SKIP_ANY = {"__pycache__", ".pytest_cache", ".ruff_cache"}
TEXT_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".md", ".txt", ".cfg", ".ini", ".sh", ".json", ""}
SELF = Path(__file__).resolve()


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        parts = path.relative_to(root).parts
        if parts[0] in SKIP_TOP or any(part in SKIP_ANY for part in parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES and path.resolve() != SELF:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="new package name (snake_case, valid Python identifier)")
    parser.add_argument("--description", help="one-sentence project description for pyproject.toml")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    name, root = args.name.strip(), args.root.resolve()
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name) or keyword.iskeyword(name):
        parser.error(f"'{name}' must be a lowercase Python identifier, e.g. my_project")
    src = root / "src" / OLD
    if not src.is_dir():
        parser.error(f"{src} not found; was the project already renamed?")
    cli_prefix = name.replace("_", "-") + "-"

    changed = []
    for path in iter_text_files(root):
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        new = re.sub(rf"\b{OLD}\b", name, text).replace(
            OLD_CLI_PREFIX + "train", cli_prefix + "train"
        )
        new = new.replace(OLD_CLI_PREFIX + "eval", cli_prefix + "eval")
        if args.description:
            new = new.replace(OLD_DESCRIPTION, args.description)
        if new != text:
            changed.append(path.relative_to(root))
            if not args.dry_run:
                path.write_text(new)

    print(f"{'Would update' if args.dry_run else 'Updated'} {len(changed)} files:")
    for p in sorted(changed):
        print(f"  {p}")
    print(f"{'Would rename' if args.dry_run else 'Renamed'} src/{OLD} -> src/{name}")
    if not args.dry_run:
        shutil.move(str(src), str(root / "src" / name))
        for egg in root.glob("src/*.egg-info"):
            shutil.rmtree(egg)
        print(
            "\nDone. Next: `uv sync` (or `mise run setup`) to reinstall the package under its new name."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
