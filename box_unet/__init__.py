""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
from box_convolution import BoxConv2d


class FeatureSampling(nn.Module):
    """ Samples features via a conv combined with box conv
    (convolution => [BN] => ReLU => 
    box convolution => [BN] => RELU"""

    def __init__(
        self,
        in_channels,
        out_channels,
        max_input_h,
        max_input_w,
        intermediate_channels_lambda=None,
        reparam_factor=1.5625,
    ):
        super().__init__()

        if intermediate_channels_lambda is None:
            intermediate_channels = max(in_channels // 4, 1)
        else:
            intermediate_channels = intermediate_channels_lambda(in_channels)

        num_box_filters = out_channels // intermediate_channels

        self.sample_features = nn.Sequential(
            nn.Conv2d(
                in_channels, intermediate_channels, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            BoxConv2d(
                intermediate_channels,
                num_box_filters,
                max_input_h,
                max_input_w,
                reparametrization_factor=reparam_factor,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.sample_features(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        max_input_h,
        max_input_w,
        intermediate_channels=None,
        reparam_factor=1.5625,
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            FeatureSampling(
                in_channels,
                out_channels,
                max_input_h,
                max_input_w,
                intermediate_channels,
                reparam_factor=reparam_factor,
            ),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        max_input_h,
        max_input_w,
        intermediate_channels=None,
        reparam_factor=1.5625,
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = FeatureSampling(
            in_channels,
            out_channels,
            max_input_h,
            max_input_w,
            intermediate_channels,
            reparam_factor=reparam_factor,
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        max_input_h,
        max_input_w,
        n_boxes=4,
        reparam_factor=1.5625,
    ):
        super().__init__()
        bt_channels = in_channels // n_boxes
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, bt_channels, (1, 1), bias=False),
            nn.BatchNorm2d(bt_channels),
            nn.ReLU(True),
            BoxConv2d(
                bt_channels,
                n_boxes,
                max_input_h,
                max_input_w,
                reparametrization_factor=reparam_factor,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class BoxUNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        max_input_h,
        max_input_w,
        intermediate_channels_lambda=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = FeatureSampling(
            in_channels,
            64,
            max_input_h,
            max_input_w,
            intermediate_channels_lambda,
        )
        self.down1 = Down(
            64, 128, max_input_h, max_input_w, intermediate_channels_lambda
        )
        self.down2 = Down(
            128, 256, max_input_h, max_input_w, intermediate_channels_lambda
        )
        self.down3 = Down(
            256, 512, max_input_h, max_input_w, intermediate_channels_lambda
        )
        self.down4 = Down(
            512, 512, max_input_h, max_input_w, intermediate_channels_lambda
        )
        self.up1 = Up(
            1024, 256, max_input_h, max_input_w, intermediate_channels_lambda
        )
        self.up2 = Up(
            512, 128, max_input_h, max_input_w, intermediate_channels_lambda
        )
        self.up3 = Up(
            256, 64, max_input_h, max_input_w, intermediate_channels_lambda
        )
        self.up4 = Up(
            128, 64, max_input_h, max_input_w, intermediate_channels_lambda
        )
        self.outc = OutConv(64, out_channels, max_input_h, max_input_w)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class PlaneLoader(torch.utils.data.Dataset):
    def __init__(self, gt_data, noisy_data):
        self.gt_data = torch.load(gt_data)
        self.noisy_data = torch.load(noisy_data)

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, index):
        return (self.gt_data[index][:, :], self.noisy_data[index][:, :])

