import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class GOTDHungarianMatcher(nn.Module):
    def __init__(
        self,
        # alphas
        loss_bbox_coeff: float = 1,
        loss_giou_coeff: float = 1,
        # betas
        cost_class_coeff: float = 1,
        cost_bbox_coeff: float = 1,
        cost_gaze_watch_outside_coeff: float = 1,
        cost_gaze_heatmap_coeff: float = 1,
    ):
        super().__init__()

        assert (
            cost_class_coeff != 0
            or cost_bbox_coeff != 0
            or cost_gaze_watch_outside_coeff != 0
            or cost_gaze_heatmap_coeff != 0
        ), "all costs cant be 0"

        self.loss_bbox_coeff = loss_bbox_coeff
        self.loss_giou_coeff = loss_giou_coeff
        self.cost_class_coeff = cost_class_coeff
        self.cost_bbox_coeff = cost_bbox_coeff
        self.cost_gaze_watch_outside_coeff = cost_gaze_watch_outside_coeff
        self.cost_gaze_heatmap_coeff = cost_gaze_heatmap_coeff

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["pred_logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_heatmap = outputs["pred_gaze_heatmap"].flatten(
            0, 1
        )  # [batch_size * num_queries, gaze_heatmap_size]
        out_watch_outside = outputs["pred_gaze_watch_outside"].flatten(
            0, 1
        )  # [batch_size * num_queries, num_classes]

        # Also concat the target labels and target
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_regression_padding = torch.cat([v["regression_padding"] for v in targets])
        tgt_heatmap = torch.cat([v["gaze_heatmaps"] for v in targets]).flatten(1, 2) * (
            ~tgt_regression_padding
        )
        tgt_watch_outside = torch.cat([v["gaze_watch_outside"] for v in targets]) * (
            ~tgt_regression_padding
        )

        cost_class = -out_prob[:, tgt_ids.argmax(-1)]
        cost_bbox = self.loss_bbox_coeff * torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_gaze_heatmap = torch.cdist(out_heatmap, tgt_heatmap, p=2)
        cost_gaze_heatmap[torch.isnan(cost_gaze_heatmap)] = 0
        cost_watch_outside = torch.abs(
            torch.cdist(out_watch_outside, tgt_watch_outside.float(), p=1)
        )
        cost_giou = self.loss_giou_coeff * -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        C = (
            self.cost_bbox_coeff * (cost_giou + cost_bbox)
            + self.cost_class_coeff * cost_class
            + self.cost_gaze_watch_outside_coeff * cost_watch_outside
            + self.cost_gaze_heatmap_coeff * cost_gaze_heatmap
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
