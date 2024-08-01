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
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import os


class PointCloudDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.pcd')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pcd_path = os.path.join(self.root_dir, self.file_list[idx])
        point_cloud = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(point_cloud.points)

        if self.train:
            points = self.augment_point_cloud(points)

        if self.transform:
            points = self.transform(points)

        return torch.from_numpy(points).float()

    def augment_point_cloud(self, points):
        # Random rotation
        rotation = self.random_rotation_3d()
        points = np.dot(points, rotation)

        # Random jitter
        points += np.random.normal(0, 0.02, size=points.shape)

        return points

    @staticmethod
    def random_rotation_3d():
        angles = np.random.uniform(0, 2*np.pi, size=3)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R
