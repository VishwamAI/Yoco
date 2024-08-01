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
import cv2
import numpy as np
from torch.utils.data import Dataset
import random


class VideoDataset(Dataset):
    def __init__(
        self, video_paths, sequence_length=16, transform=None, train=True
    ):
        self.video_paths = video_paths
        self.sequence_length = sequence_length
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = self.load_video(video_path)

        if self.train:
            frames = self.apply_augmentation(frames)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)
        return frames

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) < self.sequence_length:
            frames = frames + [frames[-1]] * (
                self.sequence_length - len(frames)
            )
        elif len(frames) > self.sequence_length:
            frames = frames[: self.sequence_length]

        return frames

    def apply_augmentation(self, frames):
        # Temporal augmentation
        if random.random() < 0.5:
            frames = frames[::-1]  # Reverse the sequence

        # Spatial augmentation
        if random.random() < 0.5:
            frames = [
                np.fliplr(frame) for frame in frames
            ]  # Horizontal flip

        return frames


# Example usage:
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# video_dataset = VideoDataset(video_paths, transform=transform)
