"""端到端 Project-Blind 验证。

把同一张 DXF 复制为两个随机命名的副本（内容逐字节相同），分别跑完整两阶段
流水线（统计判型 + 类内 hybrid 检索），比较结构类型/置信度/Top-K/相似度分数。
全部一致则验证通过。

运行：python -m src.benchmark.project_blind_verify [dxf路径]
"""

import json
import os
import shutil
import sys
import tempfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pipeline.search_pipeline import SearchPipeline

DEFAULT_DXF = os.path.join(ROOT_DIR, "data", "raw", "勾庄好运来_框剪_17.dxf")


def run(dxf_path=None, topk=5, method="hybrid", save_dir=None):
    """执行端到端 Project-Blind 验证，返回是否通过。"""
    dxf_path = dxf_path or DEFAULT_DXF
    save_dir = save_dir or os.path.join(ROOT_DIR, "fig", "benchmark", "project_blind")
    os.makedirs(save_dir, exist_ok=True)

    tmp = tempfile.mkdtemp(prefix="pbv_")
    copy_a = os.path.join(tmp, "random_001.dxf")
    copy_b = os.path.join(tmp, "random_002.dxf")
    shutil.copyfile(dxf_path, copy_a)
    shutil.copyfile(dxf_path, copy_b)

    pipe = SearchPipeline()
    # 固定到 CPU：GPU 前向存在 ~1e-6 数值噪声，会干扰"逐位一致"的验收断言
    pipe.classifier.device = "cpu"
    pipe.engine.device = "cpu"
    pipe.engine.model = pipe.engine.model.to("cpu")
    ra = pipe.search(copy_a, topk=topk, method=method)
    rb = pipe.search(copy_b, topk=topk, method=method)
    shutil.rmtree(tmp, ignore_errors=True)

    checks = {
        "type_id": ra["type_id"] == rb["type_id"],
        "confidence": ra["confidence"] == rb["confidence"],
        "topk_filenames": (
            [r["filename"] for r in ra["results"]]
            == [r["filename"] for r in rb["results"]]
        ),
        "topk_scores": (
            [round(r["score"], 6) for r in ra["results"]]
            == [round(r["score"], 6) for r in rb["results"]]
        ),
    }
    ok = all(checks.values())

    print(f"\n=== End-to-End Project-Blind Verification ===")
    print(f"查询（副本 A/B，内容相同、文件名不同）: {os.path.basename(dxf_path)}")
    print(f"方法: {method}, Top-{topk}")
    for key, value in checks.items():
        print(f"  {key:16s}: {'PASS' if value else 'FAIL'}")
    print(f"  类型: {ra['type']}({ra['type_id']}) vs {rb['type']}({rb['type_id']})")
    for i, (item_a, item_b) in enumerate(zip(ra["results"], rb["results"]), 1):
        print(f"  Top{i}: {item_a['filename']}({item_a['score']:.6f}) == "
              f"{item_b['filename']}({item_b['score']:.6f})")
    print(f"==> {'PASS：检索不依赖文件名/项目名' if ok else 'FAIL'}")

    with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"ok": ok, "checks": checks}, f, ensure_ascii=False, indent=2)
    return ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(0 if run(path) else 1)
