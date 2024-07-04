import copy

from .det_db_metric import DBMetric
from .det_metric import DetMetric, DetFCEMetric
from .rec_metric import RecMetric, CNTMetric, CANMetric
from ..postprocess import build_post_process


def build_metric(config):
    """
    The entire config dict is required.
    """
    support_dict = ["DetMetric", "DBMetric", "DetFCEMetric", "RecMetric", "CNTMetric", "CANMetric"]
    if config["Architecture"]["model_type"] == "rec":
        config = config["Metric"] | {"post_processor": build_post_process(config)}
    else:
        config = config["Metric"]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, f"{module_name} not supported, metric only supports {support_dict}"
    module_class = eval(module_name)(**config)
    return module_class
