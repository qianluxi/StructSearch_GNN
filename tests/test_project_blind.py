"""Project-blind 检索测试：项目名称不参与检索，仅用于评估。"""

from datasets.schemas import parse_drawing_meta
from benchmark.project_permutation import permute_filenames


def test_parse_drawing_meta():
    meta = parse_drawing_meta("勾庄好运来_框剪_17.dxf")
    assert meta.project_id == "勾庄好运来"
    assert meta.drawing_id == "勾庄好运来_框剪_17.dxf"


def test_search_by_type_project_blind(dataset):
    from retrieval.search_engine import SearchEngine

    engine = SearchEngine()
    # 固定到 CPU，避免 CUDA 前向的微小非确定性影响排序断言
    engine.device = "cpu"
    engine.model = engine.model.to("cpu")
    data = dataset[0].clone().to("cpu")
    type_id = int(dataset[0].y.item())

    results = []
    for fake in ("zzz_fake_a.dxf", "zzz_fake_b.dxf"):
        data.filename = fake
        results.append(
            engine.search_by_type(
                data,
                type_id=type_id,
                topk=5,
                method="hybrid",
            )
        )

    assert [r["filename"] for r in results[0]["results"]] == [
        r["filename"] for r in results[1]["results"]
    ]
    assert [r["score"] for r in results[0]["results"]] == [
        r["score"] for r in results[1]["results"]
    ]


def test_classifier_project_blind(dataset):
    from classification.classifier import StructureTypeClassifier

    clf = StructureTypeClassifier()
    preds = []
    for fake in ("zzz_fake_a.dxf", "zzz_fake_b.dxf"):
        data = dataset[0].clone()
        data.filename = fake
        preds.append(clf.predict(data))

    assert preds[0]["type_id"] == preds[1]["type_id"]
    assert preds[0]["confidence"] == preds[1]["confidence"]


def test_permute_filenames_bijective_and_deterministic():
    names = ["A_1.dxf", "A_2.dxf", "B_1.dxf", "B_2.dxf", "C_1.dxf"]
    p1, mapping1 = permute_filenames(names, seed=42)
    p2, mapping2 = permute_filenames(names, seed=42)
    p3, _ = permute_filenames(names, seed=7)

    assert list(p1) == list(p2)                    # 确定性
    assert len(set(p1)) == len(p1)                 # 双射：不重
    assert {n.split("_")[0] for n in p1} == {"A", "B", "C"}  # 项目名集合不变（置换）
    assert set(mapping1.values()) == set(mapping1.keys())    # 映射是置换
    assert any(k != v for k, v in mapping1.items())  # 非恒等
    assert list(p1) != list(p3)                    # 不同种子不同
