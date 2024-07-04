import copy


def build_post_process(config):
    """
    The entire config dict is required.
    """
    from .db_postprocess import DBPostProcess
    from .east_postprocess import EASTPostProcess
    from .fce_postprocess import FCEPostProcess

    from .rec_postprocess import CTCLabelDecode, AttnLabelDecode, NRTRLabelDecode, ViTSTRLabelDecode, CANLabelDecode

    support_dict = [
        'DBPostProcess', 'EASTPostProcess', 'FCEPostProcess', 'CTCLabelDecode', 'AttnLabelDecode', 'NRTRLabelDecode',
        'ViTSTRLabelDecode', 'CANLabelDecode'
    ]
    lang = config["lang"]
    if "Decode" in config["PostProcess"]["name"]:
        config = config["PostProcess"] | {"lang": lang}
    else:
        config = config["PostProcess"]

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(f'post process only support {support_dict}, but got {module_name}')
    module_class = eval(module_name)(**config)
    return module_class
