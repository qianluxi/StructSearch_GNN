"""Project-blind 检索测试：项目名称不参与检索，仅用于评估。"""

from datasets.schemas import parse_drawing_meta


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
