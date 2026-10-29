import os
import shutil
import unittest
from pathlib import Path

TMP = Path("tmp_tests")


@unittest.skip("skip test")
class ForecastTests(unittest.TestCase):
    def setUp(self) -> None:
        os.makedirs(TMP, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(TMP, ignore_errors=True)

    def test_forecast(self):
        self.skipTest("legacy forecast integration example is disabled")
