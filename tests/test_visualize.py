"""Tests for pure functions in 07_visualize.py."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure pipeline/ is on sys.path so `from config import ...` works inside the script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

spec = importlib.util.spec_from_file_location("visualize", "pipeline/07_visualize.py")
visualize = importlib.util.module_from_spec(spec)
sys.modules["visualize"] = visualize
spec.loader.exec_module(visualize)

_license_family = visualize._license_family


class TestLicenseFamily:
    def _make_df(self, licenses):
        return pd.DataFrame({"license": licenses})

    def test_known_licenses(self):
        df = self._make_df(["MIT", "Apache-2.0", "GPL-3.0", "BSD-2-Clause"])
        result = _license_family(df)
        assert list(result) == ["MIT", "Apache", "GPL", "BSD"]

    def test_unknown_license(self):
        df = self._make_df(["SomethingWeird"])
        result = _license_family(df)
        assert list(result) == ["Unknown/None"]

    def test_empty_string(self):
        df = self._make_df([""])
        result = _license_family(df)
        assert list(result) == ["Unknown/None"]

    def test_none_value(self):
        df = self._make_df([None])
        result = _license_family(df)
        assert list(result) == ["Unknown/None"]

    def test_noassertion(self):
        df = self._make_df(["NOASSERTION"])
        result = _license_family(df)
        assert list(result) == ["Unknown/None"]

    def test_returns_numpy_array(self):
        df = self._make_df(["MIT", "Apache-2.0"])
        result = _license_family(df)
        assert isinstance(result, np.ndarray)

    def test_creative_commons(self):
        df = self._make_df(["CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0"])
        result = _license_family(df)
        assert list(result) == ["Creative Commons", "Creative Commons", "Creative Commons"]

    def test_other_permissive(self):
        df = self._make_df(["ISC", "Unlicense", "WTFPL", "Zlib"])
        result = _license_family(df)
        assert all(r == "Other Permissive" for r in result)
