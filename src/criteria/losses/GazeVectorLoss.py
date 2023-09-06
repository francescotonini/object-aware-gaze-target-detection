import torch
import torch.nn as nn


class GazeVectorLoss(nn.Module):
    def __init__(self, loss_weight: int = 1, gaze_vector_type: str = "dx_dy"):
        super().__init__()

        self.loss_weight = loss_weight

        assert gaze_vector_type in [
            "2d",
            "3d",
        ], f"Unknown gaze vector type {gaze_vector_type}"

        self.loss_fn = nn.MSELoss()

    def forward(self, outputs, targets, indices, **kwargs):
        idx = kwargs["src_permutation_idx"]

        tgt_regression_padding = torch.cat(
            [t["regression_padding"][i] for t, (_, i) in zip(targets, indices)],
            dim=0,
        ).squeeze(1)
        tgt_gaze_vectors = torch.cat(
            [t["gaze_vectors"][i] for t, (_, i) in zip(targets, indices)], dim=0
        ).mean(dim=1)[~tgt_regression_padding]

        pred_gaze_vectors = outputs["pred_gaze_vectors"][idx][~tgt_regression_padding]

        loss = self.loss_fn(
            pred_gaze_vectors,
            tgt_gaze_vectors,
        )

        return loss * self.loss_weight
