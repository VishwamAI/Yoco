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
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json


class ImageDataset(Dataset):
    def __init__(
        self, root_dir, annotation_file, transform=None, target_transform=None
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

        self.images = self.annotations["images"]
        self.categories = {
            cat["id"]: cat["name"] for cat in self.annotations["categories"]
        }

        if not self.transform:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((640, 640)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Get annotations for this image
        img_id = img_info["id"]
        annotations = [
            ann for ann in self.annotations["annotations"] if ann["image_id"] == img_id]

        # Create target tensor
        boxes = []
        labels = []
        for ann in annotations:
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def get_img_info(self, idx):
        return self.images[idx]
