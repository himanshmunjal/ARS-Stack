"""
tests/test_analyze_results.py

Unit tests for the grouping logic of the critical-difference diagram.
"""

import pandas as pd

from src.experiments.analyze_results import _cliques


def test_cliques_group_methods_within_cd():
    ranks = pd.Series({"a": 1.0, "b": 1.5, "c": 3.0, "d": 3.4, "e": 6.0})
    groups = [(g[0], g[1]) for g in _cliques(ranks, cd=1.0)]
    assert ("a", "b") in groups and ("c", "d") in groups
    assert all("e" not in g for g in groups)  # e is further than CD from everyone


def test_cliques_drop_groups_contained_in_larger_ones():
    ranks = pd.Series({"a": 1.0, "b": 1.2, "c": 1.4})
    assert [(g[0], g[1]) for g in _cliques(ranks, cd=1.0)] == [("a", "c")]
