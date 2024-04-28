# DB & DB++ algorithm (Differentiable Binarization)
import torch.nn as nn
import torch.nn.functional as F

from models.detection.db.backbones.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from models.detection.db.backbones.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, \
    deformable_resnet18, deformable_resnet50
from models.detection.db.seg_body import FPN, ASFBlock
from models.detection.db.seg_head import DBHead
from utilities.utils import Types


class DB(nn.Module):
    backbones = {
        'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]},
        'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
        'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]},
        'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]},
        'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]},
        'deformable_resnet18': {'models': deformable_resnet18, 'out': [64, 128, 256, 512]},
        'deformable_resnet50': {'models': deformable_resnet50, 'out': [256, 512, 1024, 2048]},
        'mobilenet_v3_small': {'models': mobilenet_v3_small, 'out': [16, 24, 48, 96]},
        'mobilenet_v3_large': {'models': mobilenet_v3_large, 'out': [24, 40, 160, 160]},
    }

    def __init__(self, params):
        super().__init__()
        backbone = params['backbone']
        pretrained = params['pretrained']

        backbone, backbone_out = self.backbones[backbone]["models"], self.backbones[backbone]["out"]
        self.backbone = backbone(pretrained=pretrained)

        if params['name'] == Types.db:
            self.segmentation_body = FPN(backbone_out, inner_channels=256)
        elif params['name'] == Types.db_pp:
            self.segmentation_body = ASFBlock(backbone_out, inter_channels=256)

        self.segmentation_head = DBHead(self.segmentation_body.out_channels)

    def forward(self, x):
        image_height, image_width = x.size()[2:]
        backbone_out = self.backbone(x)
        segmentation_body_out = self.segmentation_body(backbone_out)
        segmentation_head_out = self.segmentation_head(segmentation_body_out)
        y = F.interpolate(segmentation_head_out, size=(image_height, image_width), mode='bilinear', align_corners=True)
        return y
