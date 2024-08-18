import copy

from .base_model import BaseModel


def build_model(config):
    """
    The entire config dict is required.
    """
    config = copy.deepcopy(config)
    module_class = BaseModel(config)
    return module_class
