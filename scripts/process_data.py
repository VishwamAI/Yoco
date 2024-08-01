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
import os
from tqdm import tqdm
import cv2
from utils.general import (
    load_image,
    load_point_cloud,
    load_video,
    augment_2d,
    augment_3d,
)
from utils.general import load_coco_annotations, save_coco_annotations


def parse_args():
    parser = argparse.ArgumentParser(description="Process data for Yoco model")
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory containing raw data"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for processed data"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["2d", "3d", "video"],
        required=True,
        help="Processing mode",
    )
    parser.add_argument(
        "--augment", action="store_true", help="Apply data augmentation"
    )
    return parser.parse_args()


def process_2d_data(input_dir, output_dir, augment=False):
    annotations = load_coco_annotations(os.path.join(input_dir, "annotations.json"))
    for img_info in tqdm(annotations["images"], desc="Processing 2D data"):
        img_path = os.path.join(input_dir, img_info["file_name"])
        img = load_image(img_path)
        if augment:
            img = augment_2d(img)
        # Add more processing steps here (e.g., resizing, normalization)
        processed_img_path = os.path.join(output_dir, img_info["file_name"])
        cv2.imwrite(processed_img_path, img)
    save_coco_annotations(annotations, os.path.join(output_dir, "annotations.json"))


def process_3d_data(input_dir, output_dir, augment=False):
    for file_name in tqdm(os.listdir(input_dir), desc="Processing 3D data"):
        if file_name.endswith(".pcd"):
            pcd_path = os.path.join(input_dir, file_name)
            point_cloud = load_point_cloud(pcd_path)
            if augment:
                point_cloud = augment_3d(point_cloud)
            # Add more processing steps here
            processed_pcd_path = os.path.join(output_dir, file_name)
            point_cloud.save(processed_pcd_path)


def process_video_data(input_dir, output_dir, augment=False):
    for file_name in tqdm(os.listdir(input_dir), desc="Processing video data"):
        if file_name.endswith((".mp4", ".avi")):
            video_path = os.path.join(input_dir, file_name)
            frames = load_video(video_path)
            processed_frames = []
            for frame in frames:
                if augment:
                    frame = augment_2d(frame)
                # Add more processing steps here
                processed_frames.append(frame)
            processed_video_path = os.path.join(output_dir, file_name)
            # Save processed frames as a video
            # (You may need to implement a function to save frames as video)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.mode == "2d":
        process_2d_data(args.input, args.output, args.augment)
    elif args.mode == "3d":
        process_3d_data(args.input, args.output, args.augment)
    elif args.mode == "video":
        process_video_data(args.input, args.output, args.augment)


if __name__ == "__main__":
    main()
