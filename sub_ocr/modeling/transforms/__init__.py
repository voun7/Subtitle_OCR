def build_transform(config):
    from .tps import TPS
    from .stn import STN_ON
    from .tsrn import TSRN
    from .tbsrn import TBSRN

    support_dict = ['TPS', 'STN_ON', 'TSRN', 'TBSRN']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('transform only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
