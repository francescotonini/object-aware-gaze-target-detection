# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from src.utils.box_ops import box_xyxy_to_cxcywh


def crop(image, depth, target, p=0.5):
    if np.random.random_sample() > p:
        return image, depth, target

    w, h = image.size
    gaze_points = target["gaze_points"]
    boxes = target["boxes"]

    x_coords = torch.cat(
        [
            gaze_points[:, :, 0].flatten(),  # x
            boxes[:, 0],  # x_min
            boxes[:, 2],  # x_max
        ],
        dim=0,
    )
    y_coords = torch.cat(
        [
            gaze_points[:, :, 1].flatten(),  # y
            boxes[:, 1],  # y_min
            boxes[:, 3],  # y_max
        ],
        dim=0,
    )

    # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
    # crop_x_min = np.min( [gaze_x * width, x_min, x_max])
    crop_x_min = x_coords.min().item()

    # crop_y_min = np.min([gaze_y * height, y_min, y_max])
    crop_y_min = y_coords.min().item()

    # crop_x_max = np.max([gaze_x * width, x_min, x_max])
    crop_x_max = x_coords.max().item()

    # crop_y_max = np.max([gaze_y * height, y_min, y_max])
    crop_y_max = y_coords.max().item()

    # Randomly select a random top left corner
    if crop_x_min >= 0:
        crop_x_min = np.random.uniform(0, crop_x_min)
    if crop_y_min >= 0:
        crop_y_min = np.random.uniform(0, crop_y_min)

    # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
    crop_width_min = crop_x_max - crop_x_min
    crop_height_min = crop_y_max - crop_y_min
    crop_width_max = w - crop_x_min
    crop_height_max = h - crop_y_min

    # Randomly select a width and a height
    crop_width = np.random.uniform(crop_width_min, crop_width_max)
    crop_height = np.random.uniform(crop_height_min, crop_height_max)

    # Crop it
    cropped_image = F.crop(image, crop_y_min, crop_x_min, crop_height, crop_width)
    cropped_depth = F.crop(depth, crop_y_min, crop_x_min, crop_height, crop_width)

    # Record the crop's (x, y) offset
    offset_x, offset_y = crop_x_min, crop_y_min

    target = target.copy()

    # Convert coordinates into the cropped frame
    for box_name in ["boxes"]:
        if box_name in target:
            boxes = target[box_name]
            boxes = boxes - torch.FloatTensor(
                [offset_x, offset_y, offset_x, offset_y]
            ).to(boxes.device)
            target[box_name] = boxes

    for point_name in ["gaze_points"]:
        if point_name in target:
            points = target[point_name]
            points = points - torch.FloatTensor([offset_x, offset_y]).to(points.device)
            target[point_name] = points

    h, w = crop_height, crop_width
    target["img_size"] = torch.tensor([h, w])

    return cropped_image, cropped_depth, target


def resize(image, depth, target, size, max_size=None):
    # Size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)
    rescaled_depth = F.resize(depth, size)

    if target is None:
        return rescaled_image, rescaled_depth, None

    ratios = tuple(
        float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size)
    )
    ratio_width, ratio_height = ratios

    target = target.copy()

    for box_name in ["boxes"]:
        if box_name in target:
            boxes = target[box_name]
            boxes = boxes * torch.FloatTensor(
                [ratio_width, ratio_height, ratio_width, ratio_height]
            ).to(boxes.device)
            target[box_name] = boxes

    for point_name in ["gaze_points"]:
        if point_name in target:
            points = target[point_name]
            points = points * torch.FloatTensor([ratio_width, ratio_height]).to(
                points.device
            )
            target[point_name] = points

    h, w = size
    target["img_size"] = torch.tensor([h, w])

    return rescaled_image, rescaled_depth, target


def hflip(image, depth, target):
    flipped_image = F.hflip(image)
    flipped_depth = F.hflip(depth)

    w, h = image.size

    target = target.copy()
    for box_name in ["boxes"]:
        if box_name in target:
            boxes = target[box_name]
            boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
                [-1, 1, -1, 1]
            ) + torch.as_tensor([w, 0, w, 0])
            target[box_name] = boxes

    for point_name in ["gaze_points"]:
        if point_name in target:
            points = target[point_name]
            points = points[:, :, [0, 1]] * torch.as_tensor([-1, 1]) + torch.as_tensor(
                [w, 0]
            )
            target[point_name] = points

    return flipped_image, flipped_depth, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, depth, target=None):
        size = random.choice(self.sizes)
        return resize(img, depth, target, size, self.max_size)


class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, depth, target=None):
        return crop(img, depth, target, p=self.p)


class RandomErasing(object):
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, depth, target):
        return self.eraser(img), depth, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, depth, target):
        if random.random() < self.p:
            return hflip(img, depth, target)
        return img, depth, target


class ToTensor(object):
    def __call__(self, img, depth, target):
        return F.to_tensor(img), F.to_tensor(depth), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, depth, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, depth, None

        target = target.copy()
        h, w = image.shape[-2:]
        for box_name in ["boxes"]:
            if box_name in target:
                boxes = target[box_name]
                boxes = boxes / torch.FloatTensor([w, h, w, h]).to(boxes.device)
                boxes = box_xyxy_to_cxcywh(boxes)
                target[box_name] = boxes

        for point_name in ["gaze_points"]:
            if point_name in target:
                points = target[point_name]
                points = points / torch.FloatTensor([w, h]).to(points.device)
                target[point_name] = points

        return image, depth, target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, depth, target):
        if random.random() < self.p:
            return self.transforms1(img, depth, target)
        return self.transforms2(img, depth, target)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, depth, target):
        for t in self.transforms:
            img, depth, target = t(img, depth, target)

        return img, depth, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
