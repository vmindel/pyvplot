"""Tests for the bundled Claude Code Agent Skill and pyvplot-install-skills CLI."""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

import pyvplot as vp
from pyvplot._skills.install import bundled_skill_dir, install_skill, main

SKILL = bundled_skill_dir()


def test_skill_frontmatter_and_routes():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    front = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert front, "SKILL.md needs YAML frontmatter"
    fields = dict(line.split(": ", 1) for line in front.group(1).splitlines())
    assert fields["name"] == "pyvplot"
    assert 100 < len(fields["description"]) <= 1024

    routed = set(re.findall(r"`((?:references|scripts)/[\w.-]+)`", text))
    assert routed, "router table is empty"
    for rel in routed:
        assert (SKILL / rel).is_file(), f"SKILL.md routes to missing {rel}"

    on_disk = {f"references/{p.name}" for p in (SKILL / "references").glob("*.md")}
    assert on_disk <= routed, f"reference files not routed from SKILL.md: {on_disk - routed}"


def test_references_only_mention_real_api():
    names: set[str] = set()
    for path in [SKILL / "SKILL.md", *(SKILL / "references").glob("*.md")]:
        content = path.read_text(encoding="utf-8")
        names |= set(re.findall(r"\bvp\.([A-Za-z_]\w*)", content))
        for ref in re.findall(r"`(references/[\w.-]+)`", content):
            assert (SKILL / ref).is_file(), f"{path.name} points to missing {ref}"
    missing = sorted(n for n in names if not hasattr(vp, n))
    assert not missing, f"skill mentions vp.{missing} which pyvplot does not export"


def test_install_cli_and_overwrite(tmp_path, capsys):
    dest = install_skill(tmp_path / "skills" / "pyvplot")
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == (SKILL / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert (dest / "references" / "vplots.md").is_file()
    assert (dest / "scripts" / "check_extract.py").is_file()

    with pytest.raises(FileExistsError):
        install_skill(dest)
    install_skill(dest, force=True)

    assert main(["--dest", str(dest)]) == 1
    assert main(["--dest", str(dest), "--force"]) == 0
    assert main(["--print-path"]) == 0
    assert capsys.readouterr().out.strip().endswith(str(SKILL))


def test_check_extract_script_passes():
    script = SKILL / "scripts" / "check_extract.py"
    out = subprocess.run(
        [sys.executable, str(script), "--synthetic", "--n", "15"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "0 mismatches" in out.stdout
