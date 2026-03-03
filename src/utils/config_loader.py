import yaml
from types import SimpleNamespace


def _dict_to_namespace(d):
    """
    递归将 dict 转换为 SimpleNamespace
    支持多层嵌套访问：
    cfg.model.hidden_dim
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{
            k: _dict_to_namespace(v)
            for k, v in d.items()
        })
    elif isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    else:
        return d


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    # 递归转换 dict -> SimpleNamespace
    def dict2namespace(d):
        ns = SimpleNamespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(ns, k, dict2namespace(v))
            else:
                setattr(ns, k, v)
        return ns
    return dict2namespace(cfg_dict)
