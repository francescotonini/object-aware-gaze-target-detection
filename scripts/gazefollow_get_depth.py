import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm


@torch.no_grad()
def process(root_dir, subset):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device)
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    # Get all images
    paths = glob.glob(os.path.join(root_dir, subset, "**", "*.jpg"), recursive=True)
    paths.sort()

    for src_path in tqdm.tqdm(paths):
        img = cv2.imread(src_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = prediction.cpu().numpy()

        bits = 1
        if not np.isfinite(depth).all():
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            print("WARNING: Non-finite depth values present")

        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2 ** (8 * bits)) - 1
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)

        dst_path = os.path.join(
            root_dir,
            "depth" if subset == "train" else "depth2",
            src_path.split("/")[-2],
            src_path.split("/")[-1],
        )
        Path(os.path.dirname(dst_path)).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(dst_path, out.astype("uint8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", help="Root directory of dataset")
    args = parser.parse_args()

    print("Processing train")
    process(args.dataset_dir, "train")

    print("Processing test2")
    process(args.dataset_dir, "test2")
