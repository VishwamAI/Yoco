# VishwamAI
# Copyright (C) 2024 Kasinadhsarma
#
# This file is part of the VishwamAI project.
#
# VishwamAI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# VishwamAI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with VishwamAI. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Kasinadhsarma

import torch
import torch.nn as nn
import torch.nn.functional as F
from .yoco import YOCO


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class YOCO3D(nn.Module):
    def __init__(self, num_classes=80, in_channels=1):
        super(YOCO3D, self).__init__()
        self.yoco_2d = YOCO(num_classes)

        # Encoder (downsampling)
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)

        # Decoder (upsampling)
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)

        # 3D object detection
        self.fc_3d = nn.Linear(512 * 7 * 7 * 7, num_classes)

    def forward(self, x_2d, x_3d):
        # Process 2D input
        out_2d = self.yoco_2d(x_2d)

        # Process 3D input (detection)
        det_3d = self.detect_3d(x_3d)

        # Process 3D input (segmentation)
        seg_3d = self.segment_3d(x_3d)

        # Combine 2D and 3D outputs
        combined_output = torch.cat((out_2d, det_3d, seg_3d), dim=1)

        return combined_output

    def detect_3d(self, x):
        # 3D object detection
        x1 = self.encoder1(x)
        x2 = self.encoder2(F.max_pool3d(x1, 2))
        x3 = self.encoder3(F.max_pool3d(x2, 2))
        x4 = self.encoder4(F.max_pool3d(x3, 2))
        x = F.adaptive_avg_pool3d(x4, (7, 7, 7))
        x = x.view(x.size(0), -1)
        return self.fc_3d(x)

    def segment_3d(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(F.max_pool3d(x1, 2))
        x3 = self.encoder3(F.max_pool3d(x2, 2))
        x4 = self.encoder4(F.max_pool3d(x3, 2))

        # Decoder
        x = self.upconv3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder3(x)
        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)
        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder1(x)

        return self.final_conv(x)
