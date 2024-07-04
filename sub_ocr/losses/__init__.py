import copy

# det loss
from .det_db_loss import DBLoss
from .det_east_loss import EASTLoss
from .det_fce_loss import FCELoss
from .det_pse_loss import PSELoss
from .rec_att_loss import AttentionLoss
from .rec_can_loss import CANLoss
from .rec_ce_loss import CELoss
# rec loss
from .rec_ctc_loss import CTCLoss
from .rec_nrtr_loss import NRTRLoss
from .rec_srn_loss import SRNLoss


def build_loss(config):
    support_dict = ["DBLoss", "PSELoss", "EASTLoss", "FCELoss", "CTCLoss", "AttentionLoss", "SRNLoss", "CELoss",
                    "CANLoss", "NRTRLoss"]
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, f"loss only support {support_dict}"
    module_class = eval(module_name)(**config)
    return module_class
