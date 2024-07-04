def build_neck(config):
    from .db_fpn import DBFPN, RSEFPN, LKPAN
    from .east_fpn import EASTFPN
    from .rnn import SequenceEncoder
    from .pg_fpn import PGFPN
    from .fpn import FPN
    from .fce_fpn import FCEFPN
    support_dict = ['FPN', 'DBFPN', 'EASTFPN', 'SequenceEncoder', 'PGFPN', 'RSEFPN', 'LKPAN', 'FCEFPN']

    module_name = config.pop('name')
    assert module_name in support_dict, f'{module_name} not supported, neck only supports {support_dict}'
    module_class = eval(module_name)(**config)
    return module_class
