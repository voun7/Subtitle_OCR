def build_head(config, **kwargs):
    # det head
    from .det_db_head import DBHead, PFHeadLocal
    from .det_east_head import EASTHead
    from .det_pse_head import PSEHead
    from .det_fce_head import FCEHead

    # rec head
    from .rec_ctc_head import CTCHead
    from .rec_att_head import AttentionHead
    from .rec_srn_head import SRNHead
    from .rec_nrtr_head import Transformer

    support_dict = [
        'DBHead', 'PSEHead', 'EASTHead', 'CTCHead', 'AttentionHead', 'SRNHead', 'FCEHead', 'PFHeadLocal', 'Transformer'
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, f'{module_name} not supported, head only supports {support_dict}'
    module_class = eval(module_name)(**config, **kwargs)
    return module_class
