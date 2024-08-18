def build_head(config):
    # det head
    from .det_db_head import DBHead, PFHeadLocal

    # rec head
    from .rec_ctc_head import CTCHead

    support_dict = ['DBHead', 'CTCHead', 'PFHeadLocal']

    module_name = config.pop('name')
    assert module_name in support_dict, f'{module_name} not supported, head only supports {support_dict}'
    module_class = eval(module_name)(**config)
    return module_class
