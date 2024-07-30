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
import torch.onnx
from models.yoco import YOCO
from models.yoco_3d import YOCO3D

def parse_args():
    parser = argparse.ArgumentParser(description='Export Yoco model to ONNX or TorchScript')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained weights file')
    parser.add_argument('--output', type=str, required=True, help='Output file name')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript'], default='onnx', help='Export format')
    parser.add_argument('--dim', type=str, choices=['2d', '3d'], default='2d', help='Model dimension')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='Image size')
    return parser.parse_args()

def export_onnx(model, img_size, output_path):
    dummy_input = torch.randn(1, 3, *img_size)
    torch.onnx.export(model, dummy_input, output_path, verbose=True, opset_version=11)
    print(f'ONNX model exported to {output_path}')

def export_torchscript(model, img_size, output_path):
    dummy_input = torch.randn(1, 3, *img_size)
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, output_path)
    print(f'TorchScript model exported to {output_path}')

def main():
    args = parse_args()

    # Load the model
    if args.dim == '2d':
        model = YOCO(num_classes=80)
    else:
        model = YOCO3D(num_classes=80)

    model.load_state_dict(torch.load(args.weights))
    model.eval()

    # Export the model
    if args.format == 'onnx':
        export_onnx(model, args.img_size, args.output)
    else:
        export_torchscript(model, args.img_size, args.output)

if __name__ == '__main__':
    main()
