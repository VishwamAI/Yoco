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
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.image_dataset import ImageDataset
from datasets.point_cloud_dataset import PointCloudDataset
from models.yoco import YOCO
from models.yoco_3d import YOCO3D
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os
import sys

sys.path.append("/home/ubuntu/Yoco")


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Pad images if they have different sizes
    max_h = max([img.shape[1] for img in images])
    max_w = max([img.shape[2] for img in images])
    padded_images = torch.stack(
        [
            torch.nn.functional.pad(
                img, (0, max_w - img.shape[2], 0, max_h - img.shape[1])
            )
            for img in images
        ]
    )

    # Handle cases where there are no bounding boxes
    max_num_boxes = max([target["boxes"].shape[0] for target in targets], default=1)
    padded_boxes = torch.stack(
        [
            (
                torch.nn.functional.pad(
                    target["boxes"],
                    (0, 0, 0, max_num_boxes - target["boxes"].shape[0]),
                    value=-1,
                )
                if target["boxes"].shape[0] > 0
                else torch.full((max_num_boxes, 4), -1)
            )
            for target in targets
        ]
    )
    padded_labels = torch.stack(
        [
            (
                torch.nn.functional.pad(
                    target["labels"],
                    (0, max_num_boxes - target["labels"].shape[0]),
                    value=-1,
                )
                if target["labels"].shape[0] > 0
                else torch.full((max_num_boxes,), -1)
            )
            for target in targets
        ]
    )

    padded_targets = [
        {"boxes": boxes, "labels": labels}
        for boxes, labels in zip(padded_boxes, padded_labels)
    ]

    return padded_images, padded_targets


def parse_args():
    parser = argparse.ArgumentParser(description="Train Yoco model")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dim", type=str, choices=["2d", "3d"], required=True, help="Model dimension"
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def custom_loss_function(outputs, targets):
    bbox_pred, class_pred = outputs
    bbox_targets = [t["boxes"] for t in targets]
    class_targets = [t["labels"] for t in targets]

    # Ensure bbox_pred and bbox_targets have the same shape
    bbox_pred = (
        bbox_pred.permute(0, 2, 3, 1).contiguous().view(bbox_pred.size(0), -1, 4)
    )
    bbox_targets = torch.stack(bbox_targets)

    # Match the number of predicted boxes to the number of target boxes
    if bbox_pred.size(1) > bbox_targets.size(1):
        bbox_pred = bbox_pred[:, : bbox_targets.size(1), :]
    elif bbox_pred.size(1) < bbox_targets.size(1):
        bbox_targets = bbox_targets[:, : bbox_pred.size(1), :]

    bbox_loss = torch.nn.functional.mse_loss(bbox_pred, bbox_targets)

    # Reshape class_pred to [batch_size, -1, num_classes]
    class_pred = (
        class_pred.permute(0, 2, 3, 1)
        .contiguous()
        .view(class_pred.size(0), -1, class_pred.size(1))
    )

    # Flatten class_targets to match class_pred
    class_targets = torch.cat([ct.view(-1) for ct in class_targets])

    # Ensure class_targets are long type and within valid range
    class_targets = class_targets.long()
    class_targets = torch.clamp(class_targets, 0, class_pred.size(2) - 1)

    # Reshape class_pred for cross-entropy loss
    class_pred = class_pred.view(-1, class_pred.size(2))

    # Ensure class_targets and class_pred have the same batch size
    if class_pred.size(0) > class_targets.size(0):
        class_pred = class_pred[: class_targets.size(0), :]
    elif class_pred.size(0) < class_targets.size(0):
        class_targets = class_targets[: class_pred.size(0)]

    # Compute class loss using cross entropy
    class_loss = torch.nn.functional.cross_entropy(class_pred, class_targets)

    return bbox_loss + class_loss


def train(
    model, train_loader, val_loader, optimizer, device, config, scaler, scheduler
):
    best_val_loss = float("inf")
    patience = config["patience"]
    counter = 0
    grad_accum_steps = config["gradient_accumulation_steps"]

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        for i, batch in enumerate(
            tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        ):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = custom_loss_function(outputs, targets)

            scaler.scale(loss).backward()

            if (i + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = custom_loss_function(outputs, targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(
            f'Epoch [{epoch+1}/{config["epochs"]}], '
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Early stopping and model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), config["checkpoint_path"])
            print(f'Checkpoint saved to {config["checkpoint_path"]}')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break


def main(args):
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    if args.dim == "2d":
        train_dataset = ImageDataset(
            root_dir=config["train"],
            annotation_file=config["annotation_file"],
            transform=transform,
        )
        val_dataset = ImageDataset(
            root_dir=config["val"],
            annotation_file=config["annotation_file"],
            transform=transform,
        )
    else:
        train_dataset = PointCloudDataset(root_dir=config["train"], train=True)
        val_dataset = PointCloudDataset(root_dir=config["val"], train=False)

    # Ensure annotation_file is passed correctly
    assert "annotation_file" in config, "annotation_file not found in config"
    assert os.path.exists(
        config["annotation_file"]
    ), f"Annotation file {config['annotation_file']} does not exist"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    # Initialize model, loss, and optimizer
    if args.dim == "2d":
        model = YOCO(num_classes=80)
    else:
        model = YOCO3D(num_classes=80)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
    )

    # Training
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()
    train(model, train_loader, val_loader, optimizer, device, config, scaler, scheduler)


if __name__ == "__main__":
    args = parse_args()
    main(args)
