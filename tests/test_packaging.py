#!/usr/bin/env python3
"""
tests/test_packaging.py - The plugin must actually install.

Guards the class of breakage found in the v0.4.2 audit: `.claude-plugin/marketplace.json`
advertised the repo root as a plugin source while no `.claude-plugin/plugin.json`
existed, so a Claude Code install resolved to nothing.
"""

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import config_spec  # noqa: E402  (the machine-readable egress/config contract)


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
            ".qoder-plugin/plugin.json": _load(".qoder-plugin/plugin.json")["version"],
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
        """Anything shaped like a credential must not be tracked. The pattern list is
        config_spec's, so declaring a new key format there extends this scan.
        Test fixtures assemble their fake keys from parts for the same reason."""
        patterns = [re.compile(p) for p in config_spec.SECRET_PATTERNS]
        self.assertTrue(patterns, "config_spec.SECRET_PATTERNS must not be empty")
        offenders = []
        files = (sorted(ROOT.rglob("*.py")) + sorted(ROOT.rglob("*.json"))
                 + sorted(ROOT.rglob("*.md")))
        for path in files:
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(p.search(text) for p in patterns):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_api_key_is_read_in_exactly_one_place(self):
        """The credential surface: resolve_key() is the single reader, and no other
        module may touch the environment for it. A second reader is how a key ends up
        in a log line or an evidence file three stages away."""
        readers = []
        for path in sorted((ROOT / "scripts").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if re.search(r"os\.environ(?:\.get)?\(\s*[\"']DASHSCOPE_API_KEY", text):
                readers.append(path.name)
        self.assertEqual(readers, ["omni_client.py"], f"unexpected key readers: {readers}")

    def test_omni_meta_carries_no_credential(self):
        """The per-call provenance block lands on disk next to model output, so its
        shape is a declared allowlist rather than 'whatever we happened to add'."""
        import tempfile
        import unittest.mock as mock
        from omni_client import understand_video_segment
        fake_key = "partial-" + "x" * 4
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "seg.mp4"
            clip.write_bytes(b"\x00" * 64)          # far below the inline budget
            with mock.patch("omni_client.call_omni_chat",
                            return_value=('{"visual": {"caption": "x"}}', {"total_tokens": 7})):
                data, meta = understand_video_segment(str(clip), prompt="p", api_key=fake_key)
        self.assertEqual(data["visual"]["caption"], "x")
        self.assertEqual(set(meta), {"backend", "model", "video_encoding", "endpoint_host", "usage"})
        blob = json.dumps(meta, ensure_ascii=False)
        self.assertNotIn("partial-", blob)
        self.assertNotIn("Authorization", blob)


class TestEgressDeclaration(unittest.TestCase):
    """SECURITY.md + config_spec.py are the operator-facing contract for the
    Qwen3.8-Omni dependency; documents that drift from code are how a key ends up
    being sent somewhere nobody agreed to."""

    def test_every_configured_variable_is_declared_in_all_three_documents(self):
        docs = {name: (ROOT / name).read_text(encoding="utf-8")
                for name in ("README.md", "README_CN.md", "SECURITY.md")}
        for var in config_spec.ENV_VARS:
            self.assertTrue({"name", "required", "default", "unlocks", "egress"}
                            <= set(var), f"{var.get('name')} is an incomplete spec row")
            for name, text in docs.items():
                self.assertIn(var["name"], text,
                              f"{name} must declare {var['name']} (config_spec is the source)")

    def test_declared_defaults_match_the_client_constants(self):
        import omni_client
        defaults = {var["name"]: var["default"] for var in config_spec.ENV_VARS}
        self.assertIn(omni_client.DEFAULT_MODEL, defaults["V2S_OMNI_MODEL"])
        self.assertIn(omni_client.DEFAULT_BASE_URL, defaults["DASHSCOPE_BASE_URL"])
        self.assertEqual(defaults["V2S_OMNI_ATTEMPTS"], str(omni_client.DEFAULT_ATTEMPTS))
        self.assertEqual(defaults["V2S_OMNI_TIMEOUT_SEC"], str(int(omni_client.DEFAULT_TIMEOUT_SEC)))
        self.assertEqual(config_spec.DEFAULT_BASE_URL_HOST,
                         re.match(r"https?://([^/]+)/", omni_client.DEFAULT_BASE_URL).group(1))

    def test_egress_list_names_both_omni_passes(self):
        blob = " ".join(config_spec.DATA_EGRESS)
        for script in ("speaker_diarize.py", "av_understand.py"):
            self.assertIn(script, blob, f"{script} uploads media and must be listed")

    def test_skill_frontmatter_declares_the_egress_before_activation(self):
        """The description is the one field that reaches the agent's context before the
        skill runs, so the egress disclosure has to live there, not only in a body
        section the model may never open."""
        skill = (ROOT / "skills" / "video-to-screenplay" / "SKILL.md").read_text(encoding="utf-8")
        header = skill.split("---", 2)[1]
        for needle in ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "SECURITY.md"):
            self.assertIn(needle, header, f"SKILL frontmatter must declare {needle}")


HOST_MANIFESTS = {
    "claude": ".claude-plugin/plugin.json",
    "zcode": ".zcode-plugin/plugin.json",
    "qoder": ".qoder-plugin/plugin.json",
}


class TestMultiHostPackaging(unittest.TestCase):
    """One repository, three host manifests. Field shapes genuinely differ per host
    (verified against the six packages installed under ~/.qoder-cn/plugins/cache and
    against this repo's Claude/ZCode manifests), so each host is checked against ITS
    OWN required set - not one guessed union."""

    def test_plugin_name_is_identical_and_legal(self):
        names = {_load(rel)["name"] for rel in HOST_MANIFESTS.values()}
        self.assertEqual(names, {"video-to-screenplay"})
        # Qoder's loader enforces this slug shape
        self.assertRegex("video-to-screenplay", r"^[a-z0-9]+(-[a-z0-9]+)*$")

    def test_qoder_required_fields_are_present(self):
        """All six installed Qoder packages declare exactly these four."""
        q = _load(".qoder-plugin/plugin.json")
        for field in ("name", "version", "displayName", "description"):
            self.assertTrue(str(q.get(field, "")).strip(), f"Qoder manifest needs {field}")

    def test_declared_capability_paths_exist(self):
        """A manifest pointer that resolves to nothing is how a plugin installs and
        then shows zero skills - the same failure class as the v0.4.2 marketplace bug."""
        for host, rel in HOST_MANIFESTS.items():
            manifest = _load(rel)
            declared = manifest.get("skills")
            self.assertIsNotNone(declared, f"{host} manifest must declare skills")
            for pointer in (declared if isinstance(declared, list) else [declared]):
                target = (ROOT / pointer.strip("/"))
                self.assertTrue(target.is_dir(), f"{host} skills pointer is dead: {pointer}")
                self.assertTrue(list(target.glob("*/SKILL.md")), f"{host} has no SKILL.md under {pointer}")
            commands = manifest.get("commands")
            if commands:
                cdir = ROOT / commands.strip("/")
                self.assertTrue(cdir.is_dir(), f"{host} commands pointer is dead: {commands}")
                self.assertTrue(list(cdir.glob("*.md")), f"{host} ships no command files")

    def test_command_files_carry_frontmatter(self):
        files = sorted((ROOT / "commands").glob("*.md"))
        self.assertTrue(files, "commands/ must exist for the hosts that declare it")
        for path in files:
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\n"), f"{path.name} needs frontmatter")
            front = text.split("---", 2)[1]
            fields = {line.split(":", 1)[0].strip() for line in front.splitlines() if ":" in line}
            self.assertIn("description", fields, f"{path.name} needs a description")
            self.assertIn("argument-hint", fields, f"{path.name} needs an argument-hint")

    def test_no_host_is_credited_as_author(self):
        """The ZCode manifest shipped with author "ZCode" - a plugin is authored by its
        repository owner, never by the host that runs it."""
        for host, rel in HOST_MANIFESTS.items():
            author = _load(rel).get("author")
            name = author.get("name") if isinstance(author, dict) else author
            self.assertNotIn(str(name).strip().lower(), {"zcode", "qoder", "claude", "anthropic", "openai"},
                             f"{rel} credits the host rather than the real author")
            self.assertEqual(str(name), "moyu12-ae", f"{rel} must credit the repository owner")

    def test_every_manifest_states_the_credential_it_needs(self):
        for host, rel in HOST_MANIFESTS.items():
            blob = json.dumps(_load(rel), ensure_ascii=False)
            self.assertIn("DASHSCOPE_API_KEY", blob,
                          f"{rel} must name the key the two Qwen3.8-Omni passes need")


if __name__ == "__main__":
    unittest.main()
