"""
tests/test_datasets.py

Dataset registry: core configs, the config-driven benchmark collection, and
group-name expansion used by the CLIs.
"""

import pytest

from src.data.ingest import (
    BENCHMARK_DATASETS,
    CORE_DATASETS,
    DATASET_REGISTRY,
    expand_dataset_names,
    ingest_imblearn_benchmark,
    load_dataset_config,
)


def test_core_and_benchmarks_are_registered():
    assert CORE_DATASETS == ["creditcard", "mammography", "yeast_multiclass"]
    assert len(BENCHMARK_DATASETS) == 20
    assert "mammography" not in BENCHMARK_DATASETS and "yeast_me2" not in BENCHMARK_DATASETS
    for name in BENCHMARK_DATASETS:
        assert DATASET_REGISTRY[name] is ingest_imblearn_benchmark


def test_benchmark_config_is_built_from_defaults():
    cfg = load_dataset_config("ecoli")["dataset"]
    assert cfg["type"] == "binary"
    assert cfg["target_column"] == "target" and cfg["positive_label"] == 1
    assert cfg["raw_file"] == "data/raw/benchmarks/ecoli.csv"
    assert cfg["scaling"]["columns"] == "all_numeric"


def test_core_config_still_loads_from_its_own_file():
    assert load_dataset_config("creditcard")["dataset"]["target_column"] == "Class"


def test_unknown_dataset_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset_config("not_a_dataset")


def test_expand_dataset_names():
    assert expand_dataset_names(["core"]) == CORE_DATASETS
    assert expand_dataset_names(["all"]) == CORE_DATASETS + BENCHMARK_DATASETS
    assert expand_dataset_names(["ecoli", "core", "ecoli"]) == ["ecoli"] + CORE_DATASETS
