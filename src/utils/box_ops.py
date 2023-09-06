"""
Utilities for bounding box manipulation and GIoU.
Copy-paste from DETR's official implementation with minor modifications.
"""
import torch
from torchvision.ops.boxes import box_area


def points_inside_boxes(orig_points, bboxes):
    points = orig_points.unsqueeze(1).repeat(1, len(bboxes), 1)

    if points.shape[1] != bboxes.shape[0]:
        breakpoint()

    x_min_cond = points[:, :, 0] >= bboxes[:, 0]
    y_min_cond = points[:, :, 1] >= bboxes[:, 1]
    x_max_cond = points[:, :, 0] <= bboxes[:, 2]
    y_max_cond = points[:, :, 1] <= bboxes[:, 3]

    # If point is inside bbox, then sum of above conditions will be 4
    return (
        x_min_cond.int() + y_min_cond.int() + x_max_cond.int() + y_max_cond.int()
    ) == 4


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [n,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [n,M,2]

    wh = (rb - lt).clamp(min=0)  # [n,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [n,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The target should be in [x0, y0, x1, y1] format

    Returns a [n, M] pairwise matrix, where n = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate target gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [n,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
