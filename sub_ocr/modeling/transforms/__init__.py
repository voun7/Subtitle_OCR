def build_transform(config):
    from .tps import TPS
    from .stn import STN_ON

    support_dict = ['TPS', 'STN_ON']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(f'transform only support {support_dict}')
    module_class = eval(module_name)(**config)
    return module_class
