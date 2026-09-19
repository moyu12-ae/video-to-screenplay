#!/usr/bin/env python3
"""
tests/test_packaging.py - The plugin must actually install.

Guards the class of breakage found in the v0.4.2 audit: `.claude-plugin/marketplace.json`
advertised the repo root as a plugin source while no `.claude-plugin/plugin.json`
existed, so a Claude Code install resolved to nothing.
"""

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


class TestPluginManifests(unittest.TestCase):

    def test_marketplace_source_has_a_manifest(self):
        market = _load(".claude-plugin/marketplace.json")
        self.assertIn("plugins", market)
        for entry in market["plugins"]:
            source = entry["source"]
            self.assertTrue(source.startswith("./"),
                            f"only in-repo sources are checkable here, got {source!r}")
            manifest = (ROOT / source[2:] / ".claude-plugin" / "plugin.json").resolve()
            self.assertTrue(manifest.is_file(),
                            f"marketplace entry '{entry['name']}' points at {source} "
                            "which has no .claude-plugin/plugin.json")

    def test_plugin_name_matches_everywhere(self):
        names = {
            _load(".claude-plugin/plugin.json")["name"],
            _load(".zcode-plugin/plugin.json")["name"],
            _load(".claude-plugin/marketplace.json")["plugins"][0]["name"],
        }
        self.assertEqual(names, {"video-to-screenplay"})

    def test_versions_are_consistent(self):
        versions = {
            ".claude-plugin/plugin.json": _load(".claude-plugin/plugin.json")["version"],
            ".zcode-plugin/plugin.json": _load(".zcode-plugin/plugin.json")["version"],
            ".claude-plugin/marketplace.json": _load(".claude-plugin/marketplace.json")
            ["plugins"][0]["version"],
            ".claude-plugin/marketplace.json (metadata)": _load(
                ".claude-plugin/marketplace.json")["metadata"]["version"],
        }
        self.assertEqual(len(set(versions.values())), 1, versions)

    def test_manifest_version_matches_the_evolution_record(self):
        """The repo carries one version line: EVOLUTION.md's newest section is the
        release, and the host manifests must say the same number. (They drifted to
        1.0.0 while the record was at v0.4.x.)"""
        evolution = (ROOT / "EVOLUTION.md").read_text(encoding="utf-8")
        sections = re.findall(r"^##\s*\d+\.\s*v(\d+\.\d+\.\d+)", evolution, re.MULTILINE)
        self.assertTrue(sections, "EVOLUTION.md must record a versioned section per release")
        latest = sections[-1]
        for rel in (".claude-plugin/plugin.json", ".zcode-plugin/plugin.json"):
            self.assertEqual(_load(rel)["version"], latest,
                             f"{rel} says {_load(rel)['version']} but EVOLUTION.md's "
                             f"latest release is v{latest}")

    def test_skill_frontmatter_name_matches_plugin(self):
        skill = (ROOT / "skills" / "video-to-screenplay" / "SKILL.md").read_text(encoding="utf-8")
        self.assertTrue(skill.startswith("---\n"), "SKILL.md must open with frontmatter")
        front = skill.split("---", 2)[1]
        found = {line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
                 for line in front.splitlines() if ":" in line}
        self.assertEqual(found.get("name"), "video-to-screenplay")
        self.assertTrue(found.get("description"), "SKILL.md needs a description for activation")

    def test_skill_directory_is_discoverable(self):
        manifest = _load(".zcode-plugin/plugin.json")
        skills_root = ROOT / manifest.get("skills", "skills")
        self.assertTrue(skills_root.is_dir())
        self.assertTrue((skills_root / "video-to-screenplay" / "SKILL.md").is_file())

    def test_api_key_requirement_is_declared_in_both_readmes(self):
        for name in ("README.md", "README_CN.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("DASHSCOPE_API_KEY", text, f"{name} must state the key requirement")

    def test_no_secret_looking_literals_are_committed(self):
        """Anything shaped like a DashScope key must not be tracked in the repo.
        Test fixtures assemble their fake keys from parts for the same reason."""
        pattern = re.compile(r"sk-[A-Za-z0-9]{20,}")
        offenders = []
        for path in sorted(ROOT.rglob("*.py")) + sorted(ROOT.rglob("*.json")):
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
