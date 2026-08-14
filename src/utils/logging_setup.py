"""日志初始化：统一级别控制。

默认级别为 INFO，逐构件调试信息（DXF 解析、长肢厚度计算等）走 DEBUG，
默认被过滤；设置环境变量 ``STRUCTSEARCH_LOG_LEVEL=DEBUG`` 即可复现完整
verbose 输出。

handler 始终写入“当前” sys.stderr，因此与 utils.run_output 的 Tee 重定向
兼容：运行期间日志会同时进入控制台与 experiments/runs/<时间戳>_*/logs.txt。
"""

import logging
import os
import sys


DEFAULT_LOG_LEVEL = "INFO"

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _CurrentStderrHandler(logging.Handler):
    """始终写入当前 sys.stderr，兼容 run_output 的 Tee 重定向。"""

    def emit(self, record):
        try:
            sys.stderr.write(self.format(record) + "\n")
            sys.stderr.flush()
        except Exception:  # noqa: BLE001 - 日志写入失败不应中断主流程
            self.handleError(record)


def setup_logging(level=None):
    """配置根 logger（可重复调用），返回配置后的数值级别。

    Args:
        level: 日志级别名（如 "DEBUG" / "INFO" / "WARNING"）；为 None 时
            取环境变量 STRUCTSEARCH_LOG_LEVEL，再缺省为 INFO。

    Returns:
        int: 实际生效的 logging 级别。
    """
    level_name = (
        level
        or os.environ.get("STRUCTSEARCH_LOG_LEVEL")
        or DEFAULT_LOG_LEVEL
    ).upper()

    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = _CurrentStderrHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(numeric_level)
    return numeric_level
