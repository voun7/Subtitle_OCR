import copy

# det loss
from .det_db_loss import DBLoss
# rec loss
from .rec_ctc_loss import CTCLoss


def build_loss(config):
    support_dict = ["DBLoss", "CTCLoss"]
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, f"loss only support {support_dict}"
    module_class = eval(module_name)(**config)
    return module_class
