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

import argparse
import torch
from torch.utils.data import DataLoader
from models.yoco import YOCO
from models.yoco_3d import YOCO3D
from datasets.image_dataset import ImageDataset
from datasets.point_cloud_dataset import PointCloudDataset
from utils.metrics import precision_recall_f1, mean_average_precision, confusion_matrix
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Test Yoco model")
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to test data")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--dim", type=str, choices=["2d", "3d"], required=True, help="Model dimension"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    return parser.parse_args()


def test(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            inputs, targets = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    precision, recall, f1 = precision_recall_f1(all_preds, all_targets)
    mAP = mean_average_precision(all_preds, all_targets)
    cm = confusion_matrix(all_preds, all_targets, model.num_classes)

    return precision, recall, f1, mAP, cm


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load model
    if args.dim == "2d":
        model = YOCO(num_classes=80)
    else:
        model = YOCO3D(num_classes=80)
    model.load_state_dict(torch.load(args.weights))
    model.to(device)

    # Load dataset
    if args.dim == "2d":
        dataset = ImageDataset(args.data, train=False)
    else:
        dataset = PointCloudDataset(args.data, train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Run test
    precision, recall, f1, mAP, cm = test(model, dataloader, device)

    # Print results
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"mAP: {mAP:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
