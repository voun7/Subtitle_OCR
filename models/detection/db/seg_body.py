import torch
import torch.nn.functional as F
from torch import nn


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


class ScaleChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.fc2 = nn.Conv2d(out_planes, num_features, 1, bias=False)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = self.avgpool(x)
        global_x = self.fc1(global_x)
        global_x = F.relu(self.bn(global_x))
        global_x = self.fc2(global_x)
        global_x = F.softmax(global_x, 1)
        return global_x


class ScaleChannelSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super().__init__()
        self.channel_wise = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            # nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(out_planes, in_planes, 1, bias=False)
        )
        self.spatial_wise = nn.Sequential(
            # Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_planes, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shape Nx4x1x1
        global_x = self.channel_wise(x).sigmoid()
        # shape: NxCxHxW
        global_x = global_x + x
        # shape:Nx1xHxW
        x = torch.mean(global_x, dim=1, keepdim=True)
        global_x = self.spatial_wise(x) + global_x
        global_x = self.attention_wise(global_x)
        return global_x


class ScaleSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super().__init__()
        self.spatial_wise = nn.Sequential(
            # Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_planes, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = torch.mean(x, dim=1, keepdim=True)
        global_x = self.spatial_wise(global_x) + x
        global_x = self.attention_wise(global_x)
        return global_x


class ASFBlock(nn.Module):
    def __init__(self, in_channels, inter_channels, out_features_num=4, attention_type='scale_spatial'):
        """
        Adaptive Scale Fusion (ASF) block of DBNet++
        Args:
            in_channels: the number of channels in the input data
            inter_channels: the number of middle channels
            out_features_num: the number of fused stages
        """
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.type = attention_type
        if self.type == 'scale_spatial':
            self.enhanced_attention = ScaleSpatialAttention(inter_channels, inter_channels // 4, out_features_num)
        elif self.type == 'scale_channel_spatial':
            self.enhanced_attention = ScaleChannelSpatialAttention(inter_channels, inter_channels // 4,
                                                                   out_features_num)
        elif self.type == 'scale_channel':
            self.enhanced_attention = ScaleChannelAttention(inter_channels, inter_channels // 2, out_features_num)

    def _initialize_weights(self, m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif class_name.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, concat_x, features_list):
        concat_x = self.conv(concat_x)
        score = self.enhanced_attention(concat_x)
        assert len(features_list) == self.out_features_num
        if self.type not in ['scale_channel_spatial', 'scale_spatial']:
            shape = features_list[0].shape[2:]
            score = F.interpolate(score, size=shape, mode='bilinear')
        x = []
        for i in range(self.out_features_num):
            x.append(score[:, i:i + 1] * features_list[i])
        return torch.cat(x, dim=1)
