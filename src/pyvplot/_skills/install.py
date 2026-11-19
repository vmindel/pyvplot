"""Install the bundled pyvplot Agent Skill for Claude Code."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def bundled_skill_dir() -> Path:
    """Return the path to the skill directory bundled inside the installed package."""
    return Path(__file__).resolve().parent / "data"


def default_dest() -> Path:
    """Return the default user-level Claude Code skill installation path."""
    return Path.home() / ".claude" / "skills" / "pyvplot"


def install_skill(dest: Path | str | None = None, force: bool = False) -> Path:
    """Copy the bundled pyvplot skill into ``dest`` (default ``~/.claude/skills/pyvplot``)."""
    src = bundled_skill_dir()
    if not (src / "SKILL.md").is_file():
        raise FileNotFoundError(
            f"bundled skill not found at {src}; the package may be installed without its data"
        )
    target = default_dest() if dest is None else Path(dest).expanduser()
    if target.exists():
        if not force:
            raise FileExistsError(f"{target} already exists; re-run with --force to overwrite")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        target,
        ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints", ".DS_Store"),
    )
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``pyvplot-install-skills``."""
    parser = argparse.ArgumentParser(
        prog="pyvplot-install-skills",
        description="Install the pyvplot Agent Skill for Claude Code.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="destination directory (default: ~/.claude/skills/pyvplot)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing installation",
    )
    parser.add_argument(
        "--print-path",
        action="store_true",
        help="print the bundled skill directory and exit without installing",
    )
    args = parser.parse_args(argv)

    if args.print_path:
        print(bundled_skill_dir())
        return 0
    try:
        dest = install_skill(args.dest, args.force)
    except (FileExistsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Installed pyvplot skill to {dest}")
    print("It will be available to Claude Code in your next session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
