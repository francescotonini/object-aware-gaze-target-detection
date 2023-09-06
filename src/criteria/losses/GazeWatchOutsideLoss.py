import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeWatchOutsideLoss(nn.Module):
    def __init__(self, loss_weight: int = 1):
        super().__init__()

        self.loss_weight = loss_weight

    def forward(self, outputs, targets, indices, **kwargs):
        idx = kwargs["src_permutation_idx"]

        tgt_regression_padding = torch.cat(
            [t["regression_padding"][i] for t, (_, i) in zip(targets, indices)], dim=0
        ).squeeze(1)
        tgt_watch_outside = torch.cat(
            [t["gaze_watch_outside"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )[~tgt_regression_padding].flatten()

        pred_watch_outside = outputs["pred_gaze_watch_outside"][idx][
            ~tgt_regression_padding
        ].flatten()

        loss = F.binary_cross_entropy_with_logits(
            pred_watch_outside, tgt_watch_outside.float()
        )

        return loss * self.loss_weight
