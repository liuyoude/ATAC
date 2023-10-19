"""
modification made on the basis of link:https://github.com/Xiaoccer/MobileFaceNet_Pytorch
"""
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
import torchaudio


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

# Mobilefacenet_bottleneck_setting = [
#     # t, c , n ,s
#     [2, 128, 2, 2],
#     [4, 128, 2, 2],
#     [4, 128, 2, 2],
# ]

Mobilenetv2_bottleneck_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]

sec_setting = {
    10: [313, 20],
    8: [251, 16],
    6: [188, 12],
    4: [126, 8],
    2: [63, 4],
}


class MobileFaceNet(nn.Module):
    def __init__(self,
                 num_class,
                 c_dim=128,
                 kernels=20,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(2, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        # 20(10), 4(2), 8(4)
        self.linear7 = ConvBlock(512, 512, (8, kernels), 1, 0, dw=True, linear=True)
        # self.linear7 = ConvBlock(512, 512, (4, 10), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, c_dim, 1, 1, 0, linear=True)

        self.fc_out = nn.Linear(c_dim, num_class)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x, label):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        feature = x.view(x.size(0), -1)
        out = self.fc_out(feature)
        return out, feature


class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512, frames=313):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        # self.conv_extrctor = SincConv_fast(out_channels=mel_bins,
        #                                    kernel_size=win_len,
        #                                    stride=hop_len,
        #                                    padding=win_len // 2)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                nn.LayerNorm(frames),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out


class STgramMFN(nn.Module):
    def __init__(self, num_classes, n_mels,
                 c_dim=128,
                 sec=10,
                 win_len=1024,
                 hop_len=512,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 arcface=False, m=0.5, s=30, sub=1):
        super(STgramMFN, self).__init__()
        assert sec in sec_setting.keys()
        self.arcface = ArcMarginProduct(in_features=c_dim, out_features=num_classes,
                                        m=m, s=s, sub=sub) if arcface else arcface
        self.tgramnet = TgramNet(mel_bins=n_mels, win_len=win_len, hop_len=hop_len, frames=sec_setting[sec][0])
        self.mobilefacenet = MobileFaceNet(num_class=num_classes,
                                           c_dim=c_dim,
                                           kernels=sec_setting[sec][1],
                                           bottleneck_setting=bottleneck_setting)

        # self.fc_mel = nn.Sequential(
        #     nn.Linear(128*2, 128),
        #     nn.Linear(128, 128),)
        #
        # self.fc_t = nn.Sequential(
        #     nn.Linear(128 * 2, 128),
        #     nn.Linear(128, 128),)

    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)

    def forward(self, x_wav, x_mel, label):
        # x_mel_mean = x_mel.mean(dim=-1)
        # x_mel_max, _ = x_mel.max(dim=-1)
        x_wav, x_mel = x_wav.unsqueeze(1), x_mel.unsqueeze(1)
        x_t = self.tgramnet(x_wav)
        # x_t_mean = x_t.detach().mean(dim=-1)
        # x_t_max, _ = x_t.detach().max(dim=-1)
        x = torch.cat((x_mel, x_t.unsqueeze(1)), dim=1)
        out, feature = self.mobilefacenet(x, label)

        # feature fusion
        # mel_statical_feature = self.fc_mel(torch.cat((x_mel_mean, x_mel_max), dim=1))
        # t_statical_feature = self.fc_t(torch.cat((x_t_mean, x_t_max), dim=1))
        # feature = feature + mel_statical_feature + t_statical_feature

        if self.arcface:
            out = self.arcface(feature, label)
        return out, feature


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        # print(x.device, label.device, one_hot.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output



class AdaArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, sub=1, easy_margin=False):
        super(AdaArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        self.M = Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.M)

        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.M.data.clamp_(self.m, torch.pi)
        self.cos_m = torch.cos(self.M)
        self.sin_m = torch.sin(self.M)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = torch.cos(torch.pi - self.M)
        self.mm = torch.sin(torch.pi - self.M) * self.M

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        one_hot = torch.zeros(cosine.size(), device=x.device)
        # print(x.device, label.device, one_hot.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.training:
            self.M.data.clamp_(self.m, torch.pi)
            self.cos_m = torch.cos(self.M)
            self.sin_m = torch.sin(self.M)
            # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
            self.th = torch.cos(torch.pi - self.M)
            self.mm = torch.sin(torch.pi - self.M) * self.M
        cos_m = torch.matmul(one_hot, self.cos_m).reshape(-1, 1).repeat(1, self.out_features)
        sin_m = torch.matmul(one_hot, self.sin_m).reshape(-1, 1).repeat(1, self.out_features)
        th = torch.matmul(one_hot, self.th).reshape(-1, 1).repeat(1, self.out_features)
        mm = torch.matmul(one_hot, self.mm).reshape(-1, 1).repeat(1, self.out_features)
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m - sine * sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - th) > 0, phi, cosine - mm)

        one_hot_ = torch.zeros(cosine.size(), device=x.device)
        # print(x.device, label.device, one_hot.device)
        one_hot_.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot_ * phi) + ((1.0 - one_hot_) * cosine)
        output = output * self.s
        return output


