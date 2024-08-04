def build_neck(config):
    from .db_fpn import DBFPN, RSEFPN, LKPAN
    from .rnn import SequenceEncoder
    from .fpn import FPN
    support_dict = ['FPN', 'DBFPN', 'SequenceEncoder', 'PGFPN', 'RSEFPN', 'LKPAN']

    module_name = config.pop('name')
    assert module_name in support_dict, f'{module_name} not supported, neck only supports {support_dict}'
    module_class = eval(module_name)(**config)
    return module_class
