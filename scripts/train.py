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
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train Yoco model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='path to config file')
    parser.add_argument('--dim', type=str, choices=['2d', '3d'], required=True, help='Model dimension')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train(model, train_loader, val_loader, criterion, optimizer, device, config):
    best_val_loss = float('inf')
    patience = config['patience']
    counter = 0

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}'):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch+1}/{config["epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping and model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), config['checkpoint_path'])
            print(f'Checkpoint saved to {config["checkpoint_path"]}')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

def main(args):
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    if args.dim == '2d':
        train_dataset = ImageDataset(config['train'], transform=transform)
        val_dataset = ImageDataset(config['val'], transform=transform)
    else:
        train_dataset = PointCloudDataset(config['train'], train=True)
        val_dataset = PointCloudDataset(config['val'], train=False)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize model, loss, and optimizer
    if args.dim == '2d':
        model = YOCO(num_classes=80)
    else:
        model = YOCO3D(num_classes=80)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training
    train(model, train_loader, val_loader, criterion, optimizer, device, config)

if __name__ == '__main__':
    args = parse_args()
    main(args)
