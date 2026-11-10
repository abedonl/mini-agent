"""消歧评测集：规模、期望 id 合法、基线可复现且混合不低于基线。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "tests" / "data" / "memory_disambiguation.json"
FROZEN = ROOT / "tests" / "data" / "memory_disambiguation_baseline.json"


def _load_eval_module() -> Any:
    spec = importlib.util.spec_from_file_location("memory_eval", ROOT / "tools" / "memory_eval.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dataset_covers_disambiguation_cases() -> None:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    ids = {entry["id"] for entry in dataset["entries"]}
    categories = {query["category"] for query in dataset["queries"]}

    assert len(dataset["entries"]) >= 20
    assert len(dataset["queries"]) >= 20
    assert {"同名不同人", "同主体多事实", "关系推理", "无歧义反例"} <= categories
    for query in dataset["queries"]:
        assert query["expected"], query["id"]
        assert set(query["expected"]) <= ids, query["id"]
    # 同名不同人必须真的同名：至少两组重复 subject
    subjects = [entry["subject"] for entry in dataset["entries"] if entry.get("subject")]
    assert sum(1 for subject in set(subjects) if subjects.count(subject) >= 2) >= 2


def test_frozen_selects_run_by_embedder_model() -> None:
    module = _load_eval_module()
    document = json.loads(FROZEN.read_text(encoding="utf-8"))

    bge = module.select_frozen(document, "BAAI/bge-m3")
    assert bge is not None
    assert set(bge) == {"baseline", "hybrid"}

    offline = module.select_frozen(document, "offline-char-bigram-1024")
    assert offline is not None
    assert offline["hybrid"]["disambiguation"] == 0.962963

    assert module.select_frozen(document, "unknown-model") is None


def test_eval_reproduces_frozen_baseline_and_hybrid_is_not_worse() -> None:
    module = _load_eval_module()
    dataset = module.load_dataset(DATASET)
    result = module.run_evaluation(dataset, module.LocalHashEmbedder())

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["frozen"]
    assert module.compare_with_frozen(result, frozen) == []

    baseline = result["modes"]["baseline"]
    hybrid = result["modes"]["hybrid"]
    assert hybrid["disambiguation"] >= baseline["disambiguation"]
    assert hybrid["recall@3"] >= baseline["recall@3"]
    assert hybrid["recall@5"] >= baseline["recall@5"]
