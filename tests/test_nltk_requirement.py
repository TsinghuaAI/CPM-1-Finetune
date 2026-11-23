"""Defensive requirement-floor regression; no model or corpus downloads."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NltkRequirementTests(unittest.TestCase):
    def test_floor_excludes_cve_2019_14751_versions(self):
        # Upstream fixed the downloader Zip Slip bug in NLTK 3.4.5:
        # https://github.com/nltk/nltk/commit/f59d7ed8df2e0e957f7f247fe218032abdbe9a10
        requirements = (ROOT / "requirements.txt").read_text().splitlines()
        nltk = [line.strip() for line in requirements if line.lower().startswith("nltk")]
        self.assertEqual(len(nltk), 1)
        match = re.fullmatch(r"nltk>=(\d+)\.(\d+)(?:\.(\d+))?", nltk[0])
        self.assertIsNotNone(match, "Keep an open lower bound, not an obsolete exact pin")
        floor = tuple(int(part or 0) for part in match.groups())
        self.assertGreaterEqual(floor, (3, 4, 5))


if __name__ == "__main__":
    unittest.main()
