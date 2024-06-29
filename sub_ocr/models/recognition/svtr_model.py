# Scene Text Recognition with a Single Visual Model  https://arxiv.org/pdf/2205.00159
# https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/backbones/rec_lcnetv3.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = scale_value
        self.bias = bias_value

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Act(nn.Module):
    def __init__(self, act="hswish"):
        super().__init__()
        if act == "hswish":
            self.act = nn.Hardswish()
        else:
            assert act == "relu"
            self.act = nn.ReLU()
        self.lab = LearnableAffineBlock()

    def forward(self, x):
        return self.lab(self.act(x))


class LearnableRepLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, num_conv_branches=1):
        super().__init__()
        self.is_repped = False
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = (kernel_size - 1) // 2

        self.identity = (nn.BatchNorm2d(num_features=in_channels)
                         if out_channels == in_channels and stride == 1 else None)

        self.conv_kxk = nn.ModuleList(
            [ConvBNLayer(in_channels, out_channels, kernel_size, stride, groups=groups)
             for _ in range(self.num_conv_branches)]
        )

        self.conv_1x1 = (ConvBNLayer(in_channels, out_channels, 1, stride, groups=groups)
                         if kernel_size > 1 else None)

        self.lab = LearnableAffineBlock()
        self.act = Act()

    def forward(self, x):
        # for export
        if self.is_repped:
            out = self.lab(self.reparam_conv(x))
            if self.stride != 2:
                out = self.act(out)
            return out

        out = 0
        if self.identity is not None:
            out += self.identity(x)

        if self.conv_1x1 is not None:
            out += self.conv_1x1(x)

        for conv in self.conv_kxk:
            out += conv(x)

        out = self.lab(out)
        if self.stride != 2:
            out = self.act(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = torch.multiply(identity, x)
        return x


class LCNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dw_size, use_se=False, conv_kxk_num=4):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=in_channels,
            num_conv_branches=conv_kxk_num
        )
        if use_se:
            self.se = SELayer(in_channels)
        self.pw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            num_conv_branches=conv_kxk_num
        )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class PPLCNetV3(nn.Module):
    net_config = {  # k, in_c, out_c, s, use_se
        "blocks2": [[3, 16, 32, 1, False]],
        "blocks3": [[3, 32, 64, 1, False],
                    [3, 64, 64, 1, False]],
        "blocks4": [[3, 64, 128, (2, 1), False],
                    [3, 128, 128, 1, False]],
        "blocks5": [[3, 128, 256, (1, 2), False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False]],
        "blocks6": [[5, 256, 512, (2, 1), True],
                    [5, 512, 512, 1, True],
                    [5, 512, 512, (2, 1), False],
                    [5, 512, 512, 1, False]],
    }

    def __init__(self, scale=0.95, conv_kxk_num=4):
        super().__init__()
        self.scale = scale

        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=make_divisible(16 * scale),
            kernel_size=3,
            stride=2,
        )

        self.blocks2 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks2"])
            ]
        )

        self.blocks3 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks3"])
            ]
        )

        self.blocks4 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks4"])
            ]
        )

        self.blocks5 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks5"])
            ]
        )

        self.blocks6 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks6"])
            ]
        )
        self.out_channels = make_divisible(512 * scale)

    def forward(self, x):
        out_list = []
        x = self.conv1(x)

        x = self.blocks2(x)
        x = self.blocks3(x)
        out_list.append(x)
        x = self.blocks4(x)
        out_list.append(x)
        x = self.blocks5(x)
        out_list.append(x)
        x = self.blocks6(x)
        out_list.append(x)

        if self.training:
            x = F.adaptive_avg_pool2d(x, (1, 40))
        else:
            x = F.avg_pool2d(x, (2, 2))
        return x


class Im2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        # self.out_channels = in_channels

    @staticmethod
    def forward(x):
        B, C, H, W = x.size()
        assert H == 1
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)  # (NTC)(batch, width, channels)
        return x


class CTCHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(self.in_channels * 1.0)
            nn.init.uniform_(m.weight, -stdv, stdv)
            nn.init.uniform_(m.bias, -stdv, stdv)

    def forward(self, x):
        predicts = self.fc(x)
        predicts = predicts.permute(1, 0, 2)  # B, T, C ----> T, B, C
        predicts = predicts.log_softmax(2)
        return predicts


class SVTR(nn.Module):
    def __init__(self, num_class: int = 50) -> None:
        super().__init__()
        self.backbone = PPLCNetV3()
        self.neck = Im2Seq()
        self.head = CTCHead(self.backbone.out_channels, num_class)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    test_img = torch.rand([4, 3, 48, 320])  # Batch Size, Image Channel, Image Height, Image Width
    test_model = SVTR(**{"num_class": 50})
    # test_model.eval()
    test_output = test_model(test_img)
    print(test_model), print(test_output), print(test_output.shape)
