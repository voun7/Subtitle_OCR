# Differentiable Binarization algorithm (Segmentation-based Text Detection)
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.detection.db.backbones.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from models.detection.db.backbones.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, \
    deformable_resnet18, deformable_resnet50


class DBHead(nn.Module):
    def __init__(self, in_channels, k=50):
        """
        Acquire the text region probability map, the text region threshold map and the binary map through calculation.
        """
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.thresh = self._init_thresh(in_channels)
        self.thresh.apply(self.weights_init)

    def forward(self, x):
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)
        if self.training:
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        else:
            y = torch.cat((shrink_maps, threshold_maps), dim=1)
        return y

    @staticmethod
    def weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif class_name.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_up_sample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_up_sample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid()
        )
        return self.thresh

    @staticmethod
    def _init_up_sample(in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)
            ]
            if out_channels == 1:
                module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True))
            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', inplace=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                              padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FPN(nn.Module):
    def __init__(self, in_channels, inner_channels=256):
        """
        Feature Pyramid Network, or FPN, is commonly used in convolutional networks to efficiently extract features of
        each dimension of an image.
        :param in_channels: Dimensions of basic network output
        """
        super().__init__()
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(in_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(in_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(in_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(in_channels[3], inner_channels, kernel_size=1, inplace=inplace)
        # Smooth layers
        self.smooth_p4 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)

        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=inplace)
        )
        self.out_channels = self.conv_out

    def forward(self, x):
        c2, c3, c4, c5 = x
        # Top-down
        p5 = self.reduce_conv_c5(c5)
        p4 = self._up_sample_add(p5, self.reduce_conv_c4(c4))
        p4 = self.smooth_p4(p4)
        p3 = self._up_sample_add(p4, self.reduce_conv_c3(c3))
        p3 = self.smooth_p3(p3)
        p2 = self._up_sample_add(p3, self.reduce_conv_c2(c2))
        p2 = self.smooth_p2(p2)

        x = self._up_sample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        return x

    @staticmethod
    def _up_sample_add(x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    @staticmethod
    def _up_sample_cat(p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        return torch.cat([p2, p3, p4, p5], dim=1)


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
        backbone, pretrained = params['backbone'], params['pretrained']
        backbone, backbone_out = self.backbones[backbone]["models"], self.backbones[backbone]["out"]
        self.backbone = backbone(pretrained=pretrained)
        self.segmentation_body = FPN(backbone_out, inner_channels=256)
        self.segmentation_head = DBHead(self.segmentation_body.out_channels)

    def forward(self, x):
        image_height, image_width = x.size()[2:]
        backbone_out = self.backbone(x)
        segmentation_body_out = self.segmentation_body(backbone_out)
        segmentation_head_out = self.segmentation_head(segmentation_body_out)
        y = F.interpolate(segmentation_head_out, size=(image_height, image_width), mode='bilinear', align_corners=True)
        return y
