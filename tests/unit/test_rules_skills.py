"""Validation script for Antigravity rules and agent skills."""

import glob
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


class TestRulesAndSkills(unittest.TestCase):
    def test_rules_syntax_and_secrets(self):
        rules = list((ROOT / ".agents" / "rules").glob("*.md"))
        self.assertGreaterEqual(len(rules), 3)

        expected_rules = ["scientific-integrity.md", "security.md", "agent-boundaries.md"]
        for er in expected_rules:
            rule_path = ROOT / ".agents" / "rules" / er
            self.assertTrue(rule_path.exists(), f"Missing expected rule: {er}")

        secret_patterns = [
            re.compile(r"AIza[0-9A-Za-z-_]{35}"),
            re.compile(r"ghp_[0-9A-Za-z]{36}"),
            re.compile(r"-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----"),
        ]

        for rf in rules:
            content = rf.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("---"), f"Rule {rf.name} missing frontmatter delimiter")
            for pattern in secret_patterns:
                self.assertIsNone(pattern.search(content), f"Secret detected in rule {rf.name}")

    def test_skills_syntax_and_secrets(self):
        expected_skills = [
            "terrain-analysis",
            "nasa-data",
            "security-review",
            "testing",
            "deploy-cloud-run",
        ]
        secret_patterns = [
            re.compile(r"AIza[0-9A-Za-z-_]{35}"),
            re.compile(r"ghp_[0-9A-Za-z]{36}"),
            re.compile(r"-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----"),
        ]

        for sk in expected_skills:
            skill_path = ROOT / ".agents" / "skills" / sk / "SKILL.md"
            self.assertTrue(skill_path.exists(), f"Missing skill file: {skill_path}")
            content = skill_path.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("---"), f"Skill {sk} missing frontmatter")
            parts = content.split("---", 2)
            self.assertGreaterEqual(len(parts), 3, f"Malformed frontmatter in {sk}")
            frontmatter = parts[1]
            self.assertIn("name:", frontmatter, f"Missing name in {sk}")
            self.assertIn("description:", frontmatter, f"Missing description in {sk}")

            for pattern in secret_patterns:
                self.assertIsNone(pattern.search(content), f"Secret detected in skill {sk}")


if __name__ == "__main__":
    unittest.main()
