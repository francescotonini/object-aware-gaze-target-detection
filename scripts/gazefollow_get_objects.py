import argparse
import os
import random
from pathlib import Path

import pandas as pd
import tqdm
from ultralytics import YOLO

random.seed(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", help="Root directory of dataset")
    parser.add_argument(
        "--subset",
        help="Subset of dataset to process",
        choices=["train", "test"],
    )
    args = parser.parse_args()

    labels_path = f"{args.subset if args.subset == 'train' else 'test2'}.csv"

    column_names = [
        "path",
        "idx",
        "body_bbox_x",
        "body_bbox_y",
        "body_bbox_w",
        "body_bbox_h",
        "eye_x",
        "eye_y",
        "gaze_x",
        "gaze_y",
        "bbox_x_min",
        "bbox_y_min",
        "bbox_x_max",
        "bbox_y_max",
    ]
    df = pd.read_csv(
        os.path.join(args.dataset_dir, labels_path),
        sep=",",
        names=column_names,
        usecols=column_names,
        index_col=False,
    )
    df = df.groupby("path")

    model = YOLO("yolov8n.pt")

    paths = list(df.groups.keys())

    csv = []
    for path in tqdm.tqdm(paths):
        folder = Path(os.path.dirname(path).split("/")[-1])

        results = model(os.path.join(args.dataset_dir, path))
        for result in results:
            boxes = result.boxes.xyxyn
            confs = result.boxes.conf
            clss = result.boxes.cls

            for box, conf, cls in zip(boxes, confs, clss):
                csv.append(
                    [
                        path,
                        conf.item(),
                        int(cls.item()),
                        box[0].item(),
                        box[1].item(),
                        box[2].item(),
                        box[3].item(),
                    ]
                )

    # Write csv to DATASET_ROOT_DIR
    df = pd.DataFrame(
        csv, columns=["path", "conf", "class", "x_min", "y_min", "x_max", "y_max"]
    )
    df.to_csv(os.path.join(args.dataset_dir, f"{args.subset}_objects.csv"), index=False)
