import torch
import torch.nn as nn


class GazeHeatmapLoss(nn.Module):
    def __init__(self, loss_weight: int = 1, gaze_heatmap_size: int = 64):
        super().__init__()

        self.loss_weight = loss_weight
        self.gaze_heatmap_size = gaze_heatmap_size

        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, outputs, targets, indices, **kwargs):
        idx = kwargs["src_permutation_idx"]

        tgt_regression_padding = torch.cat(
            [t["regression_padding"][i] for t, (_, i) in zip(targets, indices)], dim=0
        ).squeeze(1)
        tgt_watch_outside = torch.cat(
            [t["gaze_watch_outside"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )[~tgt_regression_padding]
        tgt_heatmap = torch.cat(
            [t["gaze_heatmaps"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )[~tgt_regression_padding].flatten(1, 2)
        tgt_gaze_inside = (tgt_watch_outside.argmax(-1) == 0).float()

        # If pred_gaze_heatmap is list, get the last one
        if isinstance(outputs["pred_gaze_heatmap"], list):
            pred_heatmap = torch.stack(
                [outputs["pred_gaze_heatmap"][i][j] for (i, j) in zip(idx[0], idx[1])]
            )
        else:
            pred_heatmap = outputs["pred_gaze_heatmap"][idx]

        pred_heatmap = pred_heatmap[~tgt_regression_padding]
        heatmap_loss = self.loss_fn(pred_heatmap, tgt_heatmap)
        heatmap_loss = heatmap_loss.mean(dim=1)
        heatmap_loss = torch.mul(
            heatmap_loss, tgt_gaze_inside
        )  # Zero out loss when it's out-of-frame gaze case

        if tgt_gaze_inside.sum() > 0:
            heatmap_loss = heatmap_loss.sum() / tgt_gaze_inside.sum()
        else:
            heatmap_loss = heatmap_loss.sum()

        return heatmap_loss * self.loss_weight
