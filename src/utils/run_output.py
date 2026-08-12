"""运行产物管理：日志与图形按"每次运行"归档。

每次运行（--build / --search）会在 experiments/runs/ 下生成一个
带时间戳的目录，例如::

    experiments/runs/20260812_153045_build/
        logs.txt                 # 本次运行的全部控制台输出
        figs/embeddings/         # 嵌入可视化（t-SNE / PCA / 距离矩阵）
        figs/classification/     # 分类混淆矩阵
        figs/retrieval/          # 单图检索结果对比图

用法::

    with run_output("build") as run_dir:
        do_something()
        save_fig(fig_dir(run_dir, "embeddings"), ...)
"""

import datetime
import os
import sys
from contextlib import contextmanager


def runs_root():
    """运行产物根目录（experiments/runs）。"""
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
    return os.path.join(project_root, "experiments", "runs")


def create_run_dir(action="run"):
    """创建并返回本次运行的目录：experiments/runs/<时间戳>_<动作>。"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_root(), f"{timestamp}_{action}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class _Tee:
    """把写入内容同时转发到多个流（例如控制台 + 日志文件）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        self.flush()

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:  # noqa: BLE001 - 日志写入失败不应中断主流程
                pass


@contextmanager
def run_output(action="run"):
    """上下文管理器：把运行期间 stdout/stderr 同时写入 logs.txt。

    Args:
        action: 运行动作名（如 "build" / "search"），用于目录命名。

    Yields:
        run_dir: 本次运行目录的绝对路径。
    """
    run_dir = create_run_dir(action)
    log_path = os.path.join(run_dir, "logs.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_stdout, log_file)
    sys.stderr = _Tee(old_stderr, log_file)

    try:
        print(f"[Run Output] 运行目录: {run_dir}")
        print(f"[Run Output] 控制台日志将同步保存到: {log_path}")
        yield run_dir
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = old_stdout, old_stderr
        log_file.close()

    print(f"[Run Output] 本次运行产物已保存: {run_dir}")


def fig_dir(run_dir, category):
    """返回（并创建）run_dir 下 figs/<category> 子目录。"""
    path = os.path.join(run_dir, "figs", category)
    os.makedirs(path, exist_ok=True)
    return path
