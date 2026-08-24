"""两阶段架构测试：Stage1 类型识别 + Stage2 类内检索。"""


def test_classifier_predict_structure(dataset):
    from classification.classifier import StructureTypeClassifier

    clf = StructureTypeClassifier()
    data = dataset[0]
    result = clf.predict(data)

    assert result["type_id"] in (0, 1, 2)
    assert result["type"] in ("剪力墙", "框剪", "框架")
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["stats"].shape == (11,)
    assert result["data"] is data


def test_search_by_type_filters_by_label(dataset):
    from retrieval.config import CLASS_NAMES
    from retrieval.search_engine import SearchEngine

    engine = SearchEngine()
    data = dataset[0]
    type_id = int(data.y.item())

    result = engine.search_by_type(
        data,
        type_id=type_id,
        topk=3,
        method="statistics",
    )
    assert result["type_id"] == type_id
    assert 0 < len(result["results"]) <= 3
    for item in result["results"]:
        assert item["label"] == CLASS_NAMES[type_id]
