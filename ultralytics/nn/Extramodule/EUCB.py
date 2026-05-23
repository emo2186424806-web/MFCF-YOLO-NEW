import math
from functools import partial

import torch
import torch.nn as nn
from timm.layers import trunc_normal_tf_
from timm.models import named_apply

__all__ = ["EUCB"]


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=""):
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        if scheme == "normal":
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "trunc_normal":
            trunc_normal_tf_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == "relu":
        return nn.ReLU(inplace)
    elif act == "relu6":
        return nn.ReLU6(inplace)
    elif act == "leakyrelu":
        return nn.LeakyReLU(neg_slope, inplace)
    elif act == "prelu":
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == "gelu":
        return nn.GELU()
    elif act == "hswish":
        return nn.Hardswish(inplace)
    else:
        raise NotImplementedError(f"activation layer [{act}] is not found")


def channel_shuffle(x, groups):
    if groups <= 1:
        return x
    b, c, h, w = x.size()
    if c % groups != 0:
        return x
    x = x.view(b, groups, c // groups, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x


class EUCB(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, stride=1, activation="relu"):
        super().__init__()
        self.in_channels = c1
        self.out_channels = c2
        self.shuffle_groups = gcd(c1, c2)

        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c1, c1, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            act_layer(activation, inplace=True),
        )

        self.pwc = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            act_layer(activation, inplace=True),
        )

        self.init_weights("normal")

    def init_weights(self, scheme=""):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.shuffle_groups)
        x = self.pwc(x)
        return x
