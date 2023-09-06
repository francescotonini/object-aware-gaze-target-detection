import torch
from torchmetrics import MeanMetric

from src.utils.gaze_ops import get_heatmap_peak_coords, get_l2_dist


class GazePointMinDistance:
    def __init__(self, gaze_heatmap_size: int = 64):
        super().__init__()

        self.gaze_heatmap_size = gaze_heatmap_size
        self.metric = MeanMetric()

    def reset_metrics(self):
        self.metric.reset()

    def get_metrics(self):
        return {
            "gaze_point_min_distance": self.metric.compute().item(),
        }

    @torch.no_grad()
    def __call__(self, outputs, targets, indices, **kwargs):
        # If metric is not on the same device as outputs, put it
        # on the same device as outputs
        if self.metric.device != outputs["pred_logits"].device:
            self.metric = self.metric.to(outputs["pred_logits"].device)

        idx = kwargs["src_permutation_idx"]

        tgt_regression_padding = torch.cat(
            [t["regression_padding"][i] for t, (_, i) in zip(targets, indices)], dim=0
        ).squeeze(1)
        tgt_gaze_points = torch.cat(
            [t["gaze_points"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )[~tgt_regression_padding]
        tgt_gaze_points_padding = torch.cat(
            [t["gaze_points_padding"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )[~tgt_regression_padding]
        tgt_watch_outside = torch.cat(
            [t["gaze_watch_outside"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )[~tgt_regression_padding].bool()

        pred_heatmaps = outputs["pred_gaze_heatmap"][idx].reshape(
            -1, self.gaze_heatmap_size, self.gaze_heatmap_size
        )[~tgt_regression_padding]

        for idx, (
            pred_heatmap,
            tgt_gaze_point,
            tgt_gaze_point_padding,
            tgt_watch_outside,
        ) in enumerate(
            zip(
                pred_heatmaps,
                tgt_gaze_points,
                tgt_gaze_points_padding,
                tgt_watch_outside,
            )
        ):
            if tgt_watch_outside:
                continue

            pred_gaze_x, pred_gaze_y = get_heatmap_peak_coords(pred_heatmap)
            pred_gaze_coord_norm = (
                torch.tensor(
                    [pred_gaze_x, pred_gaze_y], device=tgt_gaze_point_padding.device
                )
                / pred_heatmap.shape[0]  # NOTE: this assumes heatmap is square
            ).unsqueeze(0)

            all_distances = get_l2_dist(
                tgt_gaze_point[~tgt_gaze_point_padding], pred_gaze_coord_norm
            )

            self.metric(min(all_distances))
