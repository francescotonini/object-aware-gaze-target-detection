import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.ops import box_iou
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
)

import src.datamodules.components.transforms as T
from src.utils.box_ops import box_xyxy_to_cxcywh
from src.utils.gaze_ops import (
    get_label_map,
    get_angle_magnitude,
)
from src.utils.misc import get_annotation_id, get_annotations


class GazeFollowObjects(Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms: T.Compose,
        is_train: bool = True,
        num_queries: int = 50,
        num_classes: int = 81,
        gaze_heatmap_size: int = 64,
        gaze_heatmap_default_value: float = 0.0,
        use_aux_faces_dataset: bool = False,
        use_gaze_inside_only: bool = False,
        gaze_vector_type: str = "dx_dy",
        min_object_score: float = 0.5,
        min_aux_faces_score: float = 0.9,
        faces_bbox_overflow_coeff: float = 0.1,
    ):
        self.data_dir = data_dir
        self.transforms = transforms
        self.is_train_set = is_train
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.gaze_heatmap_size = gaze_heatmap_size
        self.gaze_heatmap_default_value = gaze_heatmap_default_value
        self.use_aux_faces_dataset = use_aux_faces_dataset
        self.min_object_score = min_object_score
        self.min_aux_faces_score = min_aux_faces_score
        # Will increase/decrease the bbox of faces by this value (%)
        self.faces_bbox_overflow_coeff = faces_bbox_overflow_coeff

        self.gaze_vector_type = gaze_vector_type
        assert gaze_vector_type in [
            "2d",
            "3d",
        ], f"Unknown gaze vector type {gaze_vector_type}"

        self._prepare_gazefollow_dataset(use_gaze_inside_only)
        self._prepare_aux_objects_dataset()
        if self.use_aux_faces_dataset:
            self._prepare_aux_faces_dataset()

        self.length = len(self.keys_gazefollow)

    def _prepare_gazefollow_dataset(self, use_gaze_inside_only: bool):
        labels_path = os.path.join(
            self.data_dir,
            "train_annotations_release.txt"
            if self.is_train_set
            else "test_annotations_release.txt",
        )

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
        if self.is_train_set:
            column_names.append("inout")
        column_names.append("orig_dataset")

        df = pd.read_csv(
            labels_path,
            sep=",",
            names=column_names,
            usecols=column_names,
            index_col=False,
        )

        # Only use "in" or "out "gaze (-1 is invalid, 0 is out gaze)
        if self.is_train_set:
            if use_gaze_inside_only:
                df = df[df["inout"] == 1]
            else:
                df = df[df["inout"] != -1]

        coords = torch.tensor(
            np.array(
                (
                    df["bbox_x_min"].values,
                    df["bbox_y_min"].values,
                    df["bbox_x_max"].values,
                    df["bbox_y_max"].values,
                )
            ).transpose(1, 0)
        )
        valid_bboxes = (coords[:, 2:] >= coords[:, :2]).all(dim=1)

        df = df.loc[valid_bboxes.tolist(), :]
        df.reset_index(inplace=True)
        df = df.groupby("path")

        self.X_gazefollow = df
        self.keys_gazefollow = list(df.groups.keys())

    def _prepare_aux_objects_dataset(self):
        labels_path = os.path.join(
            self.data_dir,
            "train_objects.csv" if self.is_train_set else "test_objects.csv",
        )

        column_names = [
            "path",
            "conf",
            "class",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ]

        df = pd.read_csv(
            labels_path,
            sep=",",
            names=column_names,
            usecols=column_names,
            skiprows=[
                0,
            ],
            index_col=False,
        )

        # Keep only objects with score > min_object_score
        df = df[df["conf"] >= self.min_object_score]

        # Drop rows with invalid bboxes
        coords = torch.tensor(
            np.array(
                (
                    df["x_min"].values,
                    df["y_min"].values,
                    df["x_max"].values,
                    df["y_max"].values,
                )
            ).transpose(1, 0)
        )
        valid_bboxes = (coords[:, 2:] >= coords[:, :2]).all(dim=1)

        df = df.loc[valid_bboxes.tolist(), :]
        df.reset_index(inplace=True)
        df = df.groupby("path")

        self.X_objects_aux = df
        self.keys_objects_aux = list(df.groups.keys())

    def _prepare_aux_faces_dataset(self):
        labels_path = os.path.join(
            self.data_dir,
            "train_heads.csv" if self.is_train_set else "test_heads.csv",
        )

        column_names = [
            "path",
            "score",
            "head_bbox_x_min",
            "head_bbox_y_min",
            "head_bbox_x_max",
            "head_bbox_y_max",
        ]

        df = pd.read_csv(
            labels_path,
            sep=",",
            names=column_names,
            usecols=column_names,
            skiprows=[
                0,
            ],
            index_col=False,
        )

        # Keep only heads with high score
        df = df[df["score"] >= self.min_aux_faces_score]

        # Drop rows with invalid bboxes
        coords = torch.tensor(
            np.array(
                (
                    df["head_bbox_x_min"].values,
                    df["head_bbox_y_min"].values,
                    df["head_bbox_x_max"].values,
                    df["head_bbox_y_max"].values,
                )
            ).transpose(1, 0)
        )
        valid_bboxes = (coords[:, 2:] >= coords[:, :2]).all(dim=1)

        df = df.loc[valid_bboxes.tolist(), :]
        df.reset_index(inplace=True)
        df = df.groupby("path")

        self.X_faces_aux = df
        self.keys_faces_aux = list(df.groups.keys())

    def __getitem__(self, index: int):
        if self.is_train_set:
            return self.__get_train_item__(index)
        else:
            return self.__get_test_item__(index)

    def __len__(self):
        return self.length

    def __get_train_item__(self, index: int):
        # Load image
        img = Image.open(os.path.join(self.data_dir, self.keys_gazefollow[index]))
        img = img.convert("RGB")
        img_width, img_height = img.size

        # Load depth
        depth = Image.open(
            os.path.join(
                self.data_dir,
                self.keys_gazefollow[index],
            )
            .replace("train", "depth")
            .replace("test2", "depth2")
        )
        depth = depth.convert("L")

        boxes = []
        labels = []
        gaze_points = []
        gaze_heatmaps = []
        gaze_watch_outside = []
        for _, row in self.X_gazefollow.get_group(
            self.keys_gazefollow[index]
        ).iterrows():
            box_x_min = row["bbox_x_min"]
            box_y_min = row["bbox_y_min"]
            box_x_max = row["bbox_x_max"]
            box_y_max = row["bbox_y_max"]

            # Expand bbox
            box_width = box_x_max - box_x_min
            box_height = box_y_max - box_y_min
            box_x_min -= box_width * self.faces_bbox_overflow_coeff
            box_y_min -= box_height * self.faces_bbox_overflow_coeff
            box_x_max += box_width * self.faces_bbox_overflow_coeff
            box_y_max += box_height * self.faces_bbox_overflow_coeff

            # Jitter
            if np.random.random_sample() <= 0.5:
                bbox_overflow_coeff = np.random.random_sample() * 0.2
                box_x_min -= box_width * bbox_overflow_coeff
                box_y_min -= box_height * bbox_overflow_coeff
                box_x_max += box_width * bbox_overflow_coeff
                box_y_max += box_height * bbox_overflow_coeff

            boxes.append(
                torch.FloatTensor([box_x_min, box_y_min, box_x_max, box_y_max])
            )
            labels.append(get_annotation_id("face"))

            # Gaze point
            gaze_x = row["gaze_x"] * img_width
            gaze_y = row["gaze_y"] * img_height
            gaze_points.append(torch.FloatTensor([gaze_x, gaze_y]).view(1, 2))

            # Gaze watch outside
            gaze_watch_outside.append(row["inout"] == 0)

        if (
            self.use_aux_faces_dataset
            and self.keys_gazefollow[index] in self.keys_faces_aux
        ):
            aux_faces_boxes = []
            for _, row in self.X_faces_aux.get_group(
                self.keys_gazefollow[index]
            ).iterrows():
                # Face bbox
                box_x_min = row["head_bbox_x_min"]
                box_y_min = row["head_bbox_y_min"]
                box_x_max = row["head_bbox_x_max"]
                box_y_max = row["head_bbox_y_max"]

                box_width = box_x_max - box_x_min
                box_height = box_y_max - box_y_min
                box_x_min -= box_width * self.faces_bbox_overflow_coeff
                box_y_min -= box_height * self.faces_bbox_overflow_coeff
                box_x_max += box_width * self.faces_bbox_overflow_coeff
                box_y_max += box_height * self.faces_bbox_overflow_coeff

                aux_faces_boxes.append(
                    torch.FloatTensor([box_x_min, box_y_min, box_x_max, box_y_max])
                )

            # Calculate iou between boxes and aux_head_boxes and remove from aux_face_boxes
            # the boxes where iou is not zero
            iou = box_iou(
                torch.stack(boxes),
                torch.stack(aux_faces_boxes),
            )
            aux_faces_boxes = [
                aux_faces_boxes[i]
                for i in range(len(aux_faces_boxes))
                if iou[:, i].max() == 0
            ]

            for i in range(min(len(aux_faces_boxes), self.num_queries - len(boxes))):
                boxes.append(aux_faces_boxes[i])
                labels.append(get_annotation_id("face"))

        if self.keys_gazefollow[index] in self.X_objects_aux.groups:
            # Sort objects by conf
            objects = self.X_objects_aux.get_group(self.keys_gazefollow[index])
            objects = objects.sort_values(by=["conf"], ascending=False)

            for _, row in objects.iterrows():
                box_x_min = row["x_min"]
                box_y_min = row["y_min"]
                box_x_max = row["x_max"]
                box_y_max = row["y_max"]

                box_x_min *= img_width
                box_y_min *= img_height
                box_x_max *= img_width
                box_y_max *= img_height

                # Check that the number of boxes is not greater than num_queries
                if len(boxes) >= self.num_queries:
                    break

                boxes.append(
                    torch.FloatTensor([box_x_min, box_y_min, box_x_max, box_y_max])
                )
                labels.append(get_annotation_id(get_annotations()[row["class"]]))

        # Random color change
        if np.random.random_sample() <= 0.5:
            img = adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
            img = adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
            img = adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        target = {
            "path": self.keys_gazefollow[index],
            "img_size": torch.FloatTensor([img_height, img_width]),
            "boxes": torch.stack(boxes),
            "labels": torch.LongTensor(labels),
            "gaze_points": torch.stack(gaze_points),
            "gaze_watch_outside": torch.BoolTensor(gaze_watch_outside).long(),
        }

        # Transform image and rescale all bounding target
        img, depth, target = self.transforms(img, depth, target)
        img_height, img_width = target["img_size"]

        # Gaze vector
        gaze_vectors = []
        for idx in range(len(target["gaze_points"])):
            head_bbox = target["boxes"][idx]
            head_center_point = head_bbox[:2]
            gaze_point = target["gaze_points"][idx]

            if self.gaze_vector_type == "2d":
                angle_magnitude = get_angle_magnitude(head_center_point, gaze_point)
                gaze_vectors.append(angle_magnitude)
            elif self.gaze_vector_type == "3d":
                # Multiply head_center_point and gaze_points by the image size and create new tensor
                head_center_point_orig_size = head_center_point.clone()
                head_center_point_orig_size = (
                    (
                        head_center_point_orig_size
                        * torch.FloatTensor([img_width, img_height])
                    )
                    .floor()
                    .long()
                )
                # Clamp to image_size - 1
                head_center_point_orig_size = torch.clamp(
                    head_center_point_orig_size,
                    min=torch.LongTensor([0, 0]),
                    max=torch.LongTensor([img_width - 1, img_height - 1]),
                )

                gaze_points_orig_size = gaze_point.clone()
                gaze_points_orig_size = (
                    (gaze_points_orig_size * torch.FloatTensor([img_width, img_height]))
                    .floor()
                    .long()
                )
                # Clamp to image_size - 1
                gaze_points_orig_size = torch.clamp(
                    gaze_points_orig_size,
                    min=torch.LongTensor([0, 0]),
                    max=torch.LongTensor([img_width - 1, img_height - 1]),
                )

                head_center_point_z = torch.FloatTensor(
                    [
                        depth[
                            :,
                            head_center_point_orig_size[1],
                            head_center_point_orig_size[0],
                        ]
                    ]
                )

                gaze_points_z = torch.stack(
                    [
                        torch.FloatTensor([depth[:, gaze_point[1], gaze_point[0]]])
                        for gaze_point in gaze_points_orig_size
                    ]
                )

                # Add the z-axis to head_center_point and gaze_points
                head_center_point = torch.cat((head_center_point, head_center_point_z))
                gaze_points = torch.cat((gaze_point, gaze_points_z), dim=-1)

                angles_magnitudes = get_angle_magnitude(
                    head_center_point, gaze_points, dimension=3
                )

                gaze_vectors.append(angles_magnitudes)

        num_boxes = len(target["boxes"])

        target["gaze_vectors"] = torch.stack(gaze_vectors)

        img_size = target["img_size"].repeat(num_boxes, 1)
        target["img_size"] = img_size

        labels = torch.nn.functional.one_hot(
            target["labels"], num_classes=self.num_classes
        )
        target["labels"] = labels

        # Represents which objects have a gaze heatmap
        regression_padding = torch.full((num_boxes, 1), True)
        regression_padding[: len(target["gaze_points"])] = False
        target["regression_padding"] = regression_padding

        gaze_points = torch.full((num_boxes, 1, 2), 0).float()
        gaze_points[: len(target["gaze_points"]), :, :] = target["gaze_points"]
        target["gaze_points"] = gaze_points

        gaze_vectors_size = 2 if self.gaze_vector_type == "2d" else 3
        gaze_vectors = torch.full((num_boxes, 1, gaze_vectors_size), 0).float()
        gaze_vectors[: len(target["gaze_vectors"]), :] = target["gaze_vectors"]
        target["gaze_vectors"] = gaze_vectors

        gaze_watch_outside = torch.full((num_boxes, 1), 0).float()
        gaze_watch_outside[: len(target["gaze_watch_outside"]), 0] = target[
            "gaze_watch_outside"
        ]
        target["gaze_watch_outside"] = gaze_watch_outside.long()

        for gaze_point, regression_padding in zip(
            target["gaze_points"], target["regression_padding"]
        ):
            gaze_x, gaze_y = gaze_point.squeeze(0)
            if not regression_padding:
                gaze_heatmap = torch.zeros(
                    (self.gaze_heatmap_size, self.gaze_heatmap_size)
                )

                gaze_heatmap = get_label_map(
                    gaze_heatmap,
                    [
                        gaze_x * self.gaze_heatmap_size,
                        gaze_y * self.gaze_heatmap_size,
                    ],
                    3,
                    pdf="Gaussian",
                )
            else:
                gaze_heatmap = torch.full(
                    (self.gaze_heatmap_size, self.gaze_heatmap_size),
                    float(self.gaze_heatmap_default_value),
                )

            gaze_heatmaps.append(gaze_heatmap)

        target["gaze_heatmaps"] = torch.stack(gaze_heatmaps)

        return img, depth, target["img_size"][0], target

    def __get_test_item__(self, index: int):
        # Load image
        img = Image.open(os.path.join(self.data_dir, self.keys_gazefollow[index]))
        img = img.convert("RGB")
        img_width, img_height = img.size

        # Load depth
        depth = Image.open(
            os.path.join(
                self.data_dir,
                self.keys_gazefollow[index],
            )
            .replace("train", "depth")
            .replace("test2", "depth2")
        )
        depth = depth.convert("L")

        boxes = []
        labels = []
        gaze_points = []
        gaze_points_padding = []
        gaze_heatmaps = []
        gaze_watch_outside = []

        # Group annotations from same scene with same person
        for _, same_person_annotations in self.X_gazefollow.get_group(
            self.keys_gazefollow[index]
        ).groupby("eye_x"):
            # Group annotations of the same person
            sp_gaze_points = []
            sp_boxes = []
            sp_gaze_inside = []
            for _, row in same_person_annotations.iterrows():
                # Load bbox
                box_x_min = row["bbox_x_min"]
                box_y_min = row["bbox_y_min"]
                box_x_max = row["bbox_x_max"]
                box_y_max = row["bbox_y_max"]

                sp_boxes.append(
                    torch.FloatTensor([box_x_min, box_y_min, box_x_max, box_y_max])
                )

                gaze_x = row["gaze_x"] * img_width
                gaze_y = row["gaze_y"] * img_height
                sp_gaze_points.append(torch.FloatTensor([gaze_x, gaze_y]))
                sp_gaze_inside.append(True)

            boxes.append(torch.FloatTensor(sp_boxes[-1]))
            labels.append(get_annotation_id("face"))

            sp_gaze_points_padded = torch.full((20, 2), -1).float()
            sp_gaze_points_padded[: len(sp_gaze_points), :] = torch.stack(
                sp_gaze_points
            )
            sp_gaze_points_padding = torch.full((20,), False)
            sp_gaze_points_padding[len(sp_gaze_points) :] = True

            gaze_points.append(sp_gaze_points_padded)
            gaze_points_padding.append(sp_gaze_points_padding)

            gaze_watch_outside.append(
                (
                    torch.BoolTensor(sp_gaze_inside).sum() < len(sp_gaze_inside) / 2
                ).item()
            )

        if self.keys_gazefollow[index] in self.X_objects_aux.groups:
            objects = self.X_objects_aux.get_group(self.keys_gazefollow[index])
            objects = objects.sort_values(by="conf", ascending=False)

            for _, row in objects.iterrows():
                box_x_min = row["x_min"]
                box_y_min = row["y_min"]
                box_x_max = row["x_max"]
                box_y_max = row["y_max"]

                box_x_min *= img_width
                box_y_min *= img_height
                box_x_max *= img_width
                box_y_max *= img_height

                # Check that the number of boxes is not greater than num_queries
                if len(boxes) >= self.num_queries:
                    break

                boxes.append(
                    torch.FloatTensor([box_x_min, box_y_min, box_x_max, box_y_max])
                )
                labels.append(get_annotation_id(get_annotations()[row["class"]]))

        target = {
            "path": self.keys_gazefollow[index],
            "img_size": torch.FloatTensor([img_height, img_width]),
            "boxes": torch.stack(boxes),
            "labels": torch.LongTensor(labels),
            "gaze_points": torch.stack(gaze_points),
            "gaze_points_padding": torch.stack(gaze_points_padding),
            "gaze_watch_outside": torch.BoolTensor(gaze_watch_outside).long(),
        }

        # Transform image and rescale all bounding target
        img, depth, target = self.transforms(
            img,
            depth,
            target,
        )
        img_height, img_width = target["img_size"]

        # Gaze vector
        gaze_vectors = []
        for idx in range(len(target["gaze_points"])):
            head_bbox = target["boxes"][idx]
            head_center_point = head_bbox[:2]
            gaze_points_padding = target["gaze_points_padding"][idx]
            gaze_points = target["gaze_points"][idx] * ~gaze_points_padding.unsqueeze(
                -1
            )

            if self.gaze_vector_type == "2d":
                angles_magnitudes = get_angle_magnitude(head_center_point, gaze_points)
                gaze_vectors.append(angles_magnitudes)
            elif self.gaze_vector_type == "3d":
                # Multiply head_center_point and gaze_points by the image size and create new tensor
                head_center_point_orig_size = head_center_point.clone()
                head_center_point_orig_size = (
                    (
                        head_center_point_orig_size
                        * torch.FloatTensor([img_width, img_height])
                    )
                    .floor()
                    .long()
                )
                # Clamp to image_size - 1
                head_center_point_orig_size = torch.clamp(
                    head_center_point_orig_size,
                    min=torch.LongTensor([0, 0]),
                    max=torch.LongTensor([img_width - 1, img_height - 1]),
                )

                gaze_points_orig_size = gaze_points.clone()
                gaze_points_orig_size = (
                    (gaze_points_orig_size * torch.FloatTensor([img_width, img_height]))
                    .floor()
                    .long()
                )
                # Clamp to image_size - 1
                gaze_points_orig_size = torch.clamp(
                    gaze_points_orig_size,
                    min=torch.LongTensor([0, 0]),
                    max=torch.LongTensor([img_width - 1, img_height - 1]),
                )

                head_center_point_z = torch.FloatTensor(
                    [
                        depth[
                            :,
                            head_center_point_orig_size[1],
                            head_center_point_orig_size[0],
                        ]
                    ]
                )

                gaze_points_z = torch.stack(
                    [
                        torch.FloatTensor([depth[:, gaze_point[1], gaze_point[0]]])
                        for gaze_point in gaze_points_orig_size
                    ]
                )

                # Add the z-axis to head_center_point and gaze_points
                head_center_point = torch.cat((head_center_point, head_center_point_z))
                gaze_points = torch.cat((gaze_points, gaze_points_z), dim=-1)

                angles_magnitudes = get_angle_magnitude(
                    head_center_point, gaze_points, dimension=3
                )

                gaze_vectors.append(angles_magnitudes)

        target["gaze_vectors"] = torch.stack(gaze_vectors)

        img_size = target["img_size"].repeat(self.num_queries, 1)
        target["img_size"] = img_size

        boxes = torch.full((self.num_queries, 4), 0).float()
        boxes[: len(target["boxes"]), :] = target["boxes"]
        boxes[len(target["boxes"]) :, :] = box_xyxy_to_cxcywh(
            torch.tensor([0, 0, 1, 1])
        )
        labels = torch.full((self.num_queries, self.num_classes), 0).float()
        labels[torch.arange(len(target["boxes"])), target["labels"]] = 1
        labels[len(target["boxes"]) :, -1] = 1

        target["boxes"] = boxes
        target["labels"] = labels

        regression_padding = torch.full((self.num_queries, 1), True)
        regression_padding[: len(target["gaze_points"])] = False
        target["regression_padding"] = regression_padding

        gaze_points = torch.full((self.num_queries, 20, 2), 0).float()
        gaze_points[: len(target["gaze_points"]), :, :] = target["gaze_points"]
        target["gaze_points"] = gaze_points

        gaze_vectors_size = 2 if self.gaze_vector_type == "2d" else 3
        gaze_vectors = torch.full(
            (self.num_queries, 20, gaze_vectors_size),
            0,
        ).float()
        gaze_vectors[: len(target["gaze_vectors"]), :] = target["gaze_vectors"]
        target["gaze_vectors"] = gaze_vectors

        gaze_points_padding = torch.full((self.num_queries, 20), False)
        gaze_points_padding[: len(target["gaze_points_padding"]), :] = target[
            "gaze_points_padding"
        ]
        target["gaze_points_padding"] = gaze_points_padding

        gaze_watch_outside = torch.full((self.num_queries, 1), 0).float()
        gaze_watch_outside[: len(gaze_watch_outside), 0] = target["gaze_watch_outside"]
        target["gaze_watch_outside"] = gaze_watch_outside.long()

        for gaze_points, gaze_point_padding, regression_padding in zip(
            target["gaze_points"],
            target["gaze_points_padding"],
            target["regression_padding"],
        ):
            if not regression_padding:
                gaze_heatmap = []

                for (gaze_x, gaze_y), gaze_padding in zip(
                    gaze_points, gaze_point_padding
                ):
                    if gaze_x == -1 or gaze_padding:
                        continue

                    gaze_heatmap.append(
                        get_label_map(
                            torch.zeros(
                                (self.gaze_heatmap_size, self.gaze_heatmap_size)
                            ),
                            [
                                gaze_x * self.gaze_heatmap_size,
                                gaze_y * self.gaze_heatmap_size,
                            ],
                            3,
                            pdf="Gaussian",
                        )
                    )

                gaze_heatmap = torch.stack(gaze_heatmap)
                gaze_heatmap = gaze_heatmap.sum(dim=0) / gaze_heatmap.sum(dim=0).max()
            else:
                gaze_heatmap = torch.full(
                    (self.gaze_heatmap_size, self.gaze_heatmap_size),
                    self.gaze_heatmap_default_value,
                )

            gaze_heatmaps.append(gaze_heatmap)

        target["gaze_heatmaps"] = torch.stack(gaze_heatmaps)

        return img, depth, target["img_size"][0], target
