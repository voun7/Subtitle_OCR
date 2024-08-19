import copy

from .det_db_metric import DBMetric
from .det_metric import DetMetric
from .rec_metric import RecMetric
from ..postprocess import build_post_process


def build_metric(config):
    """
    The entire config dict is required.
    """
    support_dict = ["DetMetric", "DBMetric", "RecMetric"]
    config = config["Metric"] | {"post_processor": build_post_process(config)}
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, f"{module_name} not supported, metric only supports {support_dict}"
    module_class = eval(module_name)(**config)
    return module_class
