import torch.nn as nn

from ..backbones import build_backbone
from ..heads import build_head
from ..necks import build_neck
from ..transforms import build_transform
from ...utils import read_chars


class BaseModel(nn.Module):

    def __init__(self, config, **kwargs):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super().__init__()
        lang = config["lang"]
        config = config["Architecture"]
        in_channels = config.get('in_channels', 3)
        model_type = config['model_type']
        # Build transform, for rec, transform can be TPS. For det, transform should be None.
        if 'Transform' not in config or config['Transform'] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config['Transform']['in_channels'] = in_channels
            self.transform = build_transform(config['Transform'])
            in_channels = self.transform.out_channels

        # build backbone, backbone is need for del and rec
        if 'Backbone' not in config or config['Backbone'] is None:
            self.use_backbone = False
        else:
            self.use_backbone = True
            config["Backbone"]['in_channels'] = in_channels
            self.backbone = build_backbone(config["Backbone"], model_type)
            in_channels = self.backbone.out_channels

        # Build neck. For rec, neck can be cnn,rnn or reshape (None). For det, neck can be FPN, BIFPN and so on.
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels

        #  Build head, head is need for det and rec.
        if 'Head' not in config or config['Head'] is None:
            self.use_head = False
        else:
            self.use_head = True
            config['Head']['in_channels'] = in_channels
            if config['model_type'] == 'rec':
                char_num = len(read_chars(lang))
                config['Head']['out_channels'] = char_num
            self.head = build_head(config["Head"], **kwargs)

        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_transform:
            x = self.transform(x)
        if self.use_backbone:
            x = self.backbone(x)
        if self.use_neck:
            x = self.neck(x)
        if self.use_head:
            x = self.head(x)
        return x
