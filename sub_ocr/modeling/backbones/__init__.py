def build_backbone(config, model_type):
    if model_type == 'det':
        from .det_mobilenet_v3 import MobileNetV3
        from .det_resnet import ResNet
        from .det_resnet_vd import ResNet_vd
        from .rec_lcnetv3 import PPLCNetV3
        from .rec_hgnet import PPHGNet_small
        support_dict = ['MobileNetV3', 'ResNet', 'ResNet_vd', 'PPLCNetV3', 'PPHGNet_small']
    elif model_type == 'rec':
        from .rec_mobilenet_v3 import MobileNetV3
        from .rec_resnet_vd import ResNet
        from .rec_mv1_enhance import MobileNetV1Enhance
        from .rec_nrtr_mtb import MTB
        from .rec_svtrnet import SVTRNet
        from .rec_vitstr import ViTSTR
        from .rec_lcnetv3 import PPLCNetV3
        from .rec_hgnet import PPHGNet_small
        support_dict = ['MobileNetV1Enhance', 'MobileNetV3', 'ResNet', 'MTB', 'SVTRNet', 'ViTSTR', 'PPLCNetV3',
                        'PPHGNet_small']
    else:
        raise NotImplementedError

    module_name = config.pop('name')
    assert module_name in support_dict, (f'{module_name} not supported, when model typs is {model_type},\n'
                                         f'backbone only support {support_dict}')
    module_class = eval(module_name)(**config)
    return module_class
