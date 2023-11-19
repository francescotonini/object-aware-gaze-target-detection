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

    labels_path = f"{args.subset}.csv"

    column_names = ["path", "xmin", "ymin", "xmax", "ymax", "gazex", "gazey"]
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

        results = model(os.path.join(args.dataset_dir, "images", path), verbose=False)
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
    df = pd.DataFrame(csv, columns=["path", "conf", "class", "x_min", "y_min", "x_max", "y_max"])
    df.to_csv(os.path.join(args.dataset_dir, f"{args.subset}_objects.csv"), index=False)
