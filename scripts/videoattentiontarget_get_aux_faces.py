import argparse
import os
import random
from pathlib import Path

import cv2
import pandas as pd
import tqdm
from retinaface.pre_trained_models import get_model

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

    model = get_model("resnet50_2020-07-20", max_size=2048, device="cuda")
    model.eval()

    paths = list(df.groups.keys())

    csv = []
    for path in tqdm.tqdm(paths):
        folder = Path(os.path.dirname(path).split("/")[-1])

        img = cv2.imread(os.path.join(args.dataset_dir, "images", path))

        annotations = model.predict_jsons(img)

        for annotation in annotations:
            if len(annotation["bbox"]) == 0:
                continue

            csv.append(
                [
                    path,
                    annotation["score"],
                    annotation["bbox"][0],
                    annotation["bbox"][1],
                    annotation["bbox"][2],
                    annotation["bbox"][3],
                ]
            )

    # Write csv
    df = pd.DataFrame(
        csv, columns=["path", "score", "x_min", "y_min", "x_max", "y_max"]
    )
    df.to_csv(os.path.join(args.dataset_dir, f"{args.subset}_head.csv"), index=False)
