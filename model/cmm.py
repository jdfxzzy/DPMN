# --------------------------------------------------------
# Complementation Modulation Module (CMM)
# Modified from MMFL (ACMMM 2020) by Zuoyan Zhao
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
from torch import nn

def get_norm(name, out_channels):
    if name == 'batch':
        norm = nn.BatchNorm2d(out_channels)
    elif name == 'instance':
        norm = nn.InstanceNorm2d(out_channels)
    else:
        norm = None
    return norm


def get_act(name):
    if name == 'relu':
        activation = nn.ReLU(inplace=False)
    elif name == 'elu':
        activation = nn.ELU(inplace=False)
    elif name == 'leaky_relu':
        activation = nn.LeakyReLU(negative_slope=0.2, inplace=False)
    elif name == 'tanh':
        activation = nn.Tanh()
    elif name == 'sigmoid':
        activation = nn.Sigmoid()
    elif name == 'gelu':
        activation = nn.GELU()
    else:
        activation = None
    return activation


class EncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization='batch', activation=None):
        super().__init__()
        layers = []
        if activation is not None:
            layers.append(get_act(activation))
        layers.append(nn.Conv2d(in_channels, in_channels, 4, 2, dilation=2, padding=3))
        if normalization is not None:
            layers.append(get_norm(normalization, in_channels))
        if activation is not None:
            layers.append(get_act(activation))
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1))
        if normalization is not None:
            layers.append(get_norm(normalization, out_channels))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization=None, activation=None):
        super().__init__()
        
        layers = []
        if activation is not None:
            layers.append(get_act(activation))
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, 3, 1, padding=1))
        if normalization is not None:
            layers.append(get_norm(normalization, out_channels))

        if activation is not None:
            layers.append(get_act(activation))
        layers.append(nn.ConvTranspose2d(out_channels, out_channels, 4, 2, padding=1))
        if normalization is not None:
            layers.append(get_norm(normalization, out_channels))
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class ComplementationModulationModule(nn.Module):
    def __init__(self, c_img=3, norm='batch', act_en='leaky_relu', act_de='relu', cnum=64):
        super().__init__()

        c_in = c_img

        self.en_1_1 = nn.Conv2d(c_in, cnum, 3, 1, padding=1)
        self.en_2_1 = EncodeBlock(cnum, cnum*2, normalization=norm, activation=act_en)
        self.en_3_1 = EncodeBlock(cnum*2, cnum*4, normalization=norm, activation=act_en)
        self.en_4_1 = EncodeBlock(cnum*4, cnum*8, normalization=norm, activation=act_en)
        self.en_5_1 = EncodeBlock(cnum*8, cnum*8, normalization=norm, activation=act_en)
        self.en_6_1 = nn.Sequential(
            get_act(act_en),
            nn.Conv2d(cnum*8, cnum*8, 4, 2, padding=1))

        self.en_1_2 = nn.Conv2d(c_in, cnum, 3, 1, padding=1)
        self.en_2_2 = EncodeBlock(cnum, cnum*2, normalization=norm, activation=act_en)
        self.en_3_2 = EncodeBlock(cnum*2, cnum*4, normalization=norm, activation=act_en)
        self.en_4_2 = EncodeBlock(cnum*4, cnum*8, normalization=norm, activation=act_en)
        self.en_5_2 = EncodeBlock(cnum*8, cnum*8, normalization=norm, activation=act_en)
        self.en_6_2 = nn.Sequential(
            get_act(act_en),
            nn.Conv2d(cnum*8, cnum*8, 4, 2, padding=1))

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(16*cnum, 4*cnum)
        self.fc_2 = nn.Linear(4*cnum, 16*cnum)

        self.de_6 = nn.Sequential(
            get_act(act_de),
            nn.ConvTranspose2d(cnum*16, cnum*8, 4, 2, padding=1),
            nn.BatchNorm2d(cnum*8))
        self.de_5 = DecodeBlock(cnum*8*3, cnum*8, normalization=norm, activation=act_de)
        self.de_4 = DecodeBlock(cnum*8*3, cnum*4, normalization=norm, activation=act_de)
        self.de_3 = DecodeBlock(cnum*4*3, cnum*2, normalization=norm, activation=act_de)
        self.de_2 = DecodeBlock(cnum*2*3, cnum, normalization=norm, activation=act_de)
        self.de_1 = nn.Sequential(
            get_act(act_de),
            nn.ConvTranspose2d(cnum*3, c_img, 3, 1, padding=1))

    def forward(self, x1, x2):
        out_1_1 = self.en_1_1(x1)
        out_2_1 = self.en_2_1(out_1_1)
        out_3_1 = self.en_3_1(out_2_1)
        out_4_1 = self.en_4_1(out_3_1)
        out_5_1 = self.en_5_1(out_4_1)
        out_6_1 = self.en_6_1(out_5_1)

        out_1_2 = self.en_1_2(x2)
        out_2_2 = self.en_2_2(out_1_2)
        out_3_2 = self.en_3_2(out_2_2)
        out_4_2 = self.en_4_2(out_3_2)
        out_5_2 = self.en_5_2(out_4_2)
        out_6_2 = self.en_6_2(out_5_2)

        out_6 = torch.cat([out_6_1, out_6_2], dim=1)
        residual = out_6
        out_6 = self.pooling(out_6)
        N, C, _, _ = out_6.size()
        out_6 = out_6.view(N, -1, C)
        out_6_fc_1 = self.fc_1(out_6)
        out_6_fc_1 = nn.ReLU(inplace=True)(out_6_fc_1)
        out_6_fc_2 = self.fc_2(out_6_fc_1)
        weight = nn.Sigmoid()(out_6_fc_2)
        weight = weight.view(N, C, 1, 1)
        out_6 = residual
        out_6 = out_6 * weight
        out_6 = out_6 + residual

        d_out_6 = self.de_6(out_6)
        d_out_6_out_5 = torch.cat([d_out_6, out_5_1, out_5_2], dim=1)
        d_out_5 = self.de_5(d_out_6_out_5)
        d_out_5_out_4 = torch.cat([d_out_5, out_4_1, out_4_2], dim=1)
        d_out_4 = self.de_4(d_out_5_out_4)
        d_out_4_out_3 = torch.cat([d_out_4, out_3_1, out_3_2], dim=1)
        d_out_3 = self.de_3(d_out_4_out_3)
        d_out_3_out_2 = torch.cat([d_out_3, out_2_1, out_2_2], dim=1)
        d_out_2 = self.de_2(d_out_3_out_2)
        d_out_2_out_1 = torch.cat([d_out_2, out_1_1, out_1_2], dim=1)
        dout_1 = self.de_1(d_out_2_out_1)

        return dout_1