import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import box_ops


class BoxesLoss(nn.Module):
    def __init__(
        self,
        loss_l1_weight: int = 1,
        loss_giou_weight: int = 1,
        loss_weight: int = 1,
    ):
        super().__init__()

        self.loss_l1_weight = loss_l1_weight
        self.loss_giou_weight = loss_giou_weight
        self.loss_weight = loss_weight

    def forward(self, outputs, targets, indices, **kwargs):
        idx = kwargs["src_permutation_idx"]
        num_boxes = kwargs["num_boxes"]

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss_giou = loss_giou.sum() / num_boxes

        loss = loss_bbox * self.loss_l1_weight + loss_giou * self.loss_giou_weight

        return loss * self.loss_weight
