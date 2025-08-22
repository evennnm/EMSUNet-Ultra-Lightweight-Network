import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class SMDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6'):
        super(SMDC, self).__init__()

        self.in_channels = in_channels // 4
        self.kernel_sizes = kernel_sizes
        self.activation = activation

        self.dwconvs1 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_sizes[0], stride, kernel_sizes[0] // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True),
            )

        self.dwconvs2 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_sizes[1], stride, kernel_sizes[1] // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True),
            )
        self.dwconvs3 = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_sizes[2], stride, kernel_sizes[2] // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True),
            )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        channels_per_group = x.size(1) // 4

        group1, group2,group3, group4 = torch.split(x, [channels_per_group, channels_per_group,channels_per_group, channels_per_group], dim=1)
        group1 = self.dwconvs3(group1)
        group2 = self.dwconvs2(group2)
        group3 = self.dwconvs1(group3)

        out = torch.cat([group1, group2,group3, group4], dim=1)

        return out
class SMCA(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[3, 5, 7], expansion_factor=2,  activation='relu6'):
        super(SMCA, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.activation = activation

        assert self.stride in [1, 2]

        self.use_skip_connection = True if self.stride == 1 else False

        self.hidden_channels= expansion_factor * self.in_channels

        self.smdc = SMDC(self.hidden_channels, self.kernel_sizes, self.stride, self.activation)
        self.SE = SqueezeExcite(self.hidden_channels, 0.25)

        self.Ppconv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_channels),
            act_layer(self.activation, inplace=True)
        )
        self.Ppconv2 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):

        group2 = self.Ppconv1(x)
        group2 = self.smdc(group2)
        group2 = self.SE(group2)

        out = self.Ppconv2(group2)

        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def SMCALayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[3, 5, 7], expansion_factor=2, activation='relu6'):
    convs = []
    smca = SMCA(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, activation=activation)
    convs.append(smca)
    if n > 1:
        for i in range(1, n):
            mscb = SMCA(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,activation=activation)
            convs.append(mscb)
    conv = nn.Sequential(*convs)
    return conv

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class EMUB(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(EMUB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            SMDC(self.in_channels, [3,5,7], 1, activation,),
        )
        # self.se = SqueezeExcite(self.in_channels, 0.25)
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        # x = self.se(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


class EMAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, activation='relu'):
        super(EMAG, self).__init__()

        self.W_g = nn.Sequential(
            SMDC(F_g, [3, 5, 7], 1, activation, ),
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0,
                          bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            SMDC(F_l, [3, 5, 7], 1, activation, ),
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0,
                          bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu', init_z=0.5):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels
        self.Z = nn.Parameter(torch.tensor(init_z))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):

        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out)+1

class SAB(nn.Module):
    def __init__(self, kernel_size=3):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'

        num_conv_layers = (kernel_size - 1) // 2


        self.convs = nn.ModuleList([
            nn.Conv2d(2 if i == 0 else 1, 1, 3, padding=1, bias=False)
            for i in range(num_conv_layers)
        ])

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        for conv in self.convs:
            x = F.relu(x)
            x = conv(x)
        return self.sigmoid(x)+1

class decoder_EMSUnet(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], kernel_sizes=[3, 3, 5], expansion_factor=6, lgag_ks=3, activation='relu6'):
        super(decoder_EMSUnet, self).__init__()
        eucb_ks = 3  # kernel size for eucb
        self.SMCA4 = SMCALayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor,activation=activation)

        self.emub3 = EMUB(in_channels=channels[0], out_channels=channels[1])
        self.emag3 = EMAG(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2, kernel_size=lgag_ks)
        self.SMCA3 = SMCALayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, activation=activation)

        self.emub2 = EMUB(in_channels=channels[1], out_channels=channels[2])
        self.emag2 = EMAG(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2, kernel_size=lgag_ks)
        self.SMCA2 = SMCALayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor,activation=activation)

        self.emub1 = EMUB(in_channels=channels[2], out_channels=channels[3])
        self.emag1 = EMAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2), kernel_size=lgag_ks)
        self.SMCA1 = SMCALayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor,activation=activation)

        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])

        self.sab = SAB()

    def forward(self, x, skips):
        d4 = self.cab4(x) * x
        d4 = self.sab(d4) * d4
        d4 = self.SMCA4(d4)

        d3 = self.emub3(d4)
        x3 = self.emag3(g=d3, x=skips[0])
        d3 = d3 + x3

        d3 = self.cab3(d3) * d3
        d3 = self.sab(d3) * d3
        d3 = self.SMCA3(d3)


        d2 = self.emub2(d3)
        x2 = self.emag2(g=d2, x=skips[1])
        d2 = d2 + x2

        d2 = self.cab2(d2) * d2
        d2 = self.sab(d2) * d2
        d2 = self.SMCA2(d2)


        d1 = self.emub1(d2)
        x1 = self.emag1(g=d1, x=skips[2])
        d1 = d1 + x1

        d1 = self.cab1(d1) * d1
        d1 = self.sab(d1) * d1
        d1 = self.SMCA1(d1)


        return [d4, d3, d2, d1]
