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

class YOCORobotics(nn.Module):
    def __init__(self, num_classes=80):
        super(YOCORobotics, self).__init__()
        self.yoco_2d = YOCO(num_classes)
        self.lidar_conv = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.camera_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.fc_fusion = nn.Linear(128 * 7 * 7 * 7 + 128 * 7 * 7, num_classes)

    def forward(self, x_camera, x_lidar):
        # Process camera input
        out_camera = self.camera_process(x_camera)

        # Process LiDAR input
        out_lidar = self.lidar_process(x_lidar)

        # Sensor fusion
        fused_output = self.sensor_fusion(out_camera, out_lidar)

        return fused_output

    def camera_process(self, x):
        x = self.camera_conv(x)
        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = x.view(x.size(0), -1)
        return x

    def lidar_process(self, x):
        x = self.lidar_conv(x)
        x = F.adaptive_avg_pool3d(x, (7, 7, 7))
        x = x.view(x.size(0), -1)
        return x

    def sensor_fusion(self, x_camera, x_lidar):
        x = torch.cat((x_camera, x_lidar), dim=1)
        return self.fc_fusion(x)
