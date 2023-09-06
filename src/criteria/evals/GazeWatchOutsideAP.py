import torch
from torchmetrics import AveragePrecision


class GazeWatchOutsideAP:
    def __init__(self):
        super().__init__()

        self.metric = AveragePrecision(task="binary")

    def reset_metrics(self):
        self.metric.reset()

    def get_metrics(self):
        return {
            "gaze_watch_outside_ap": self.metric.compute().item(),
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

        # Gaze inside
        pred_watch_outside = outputs["pred_gaze_watch_outside"][idx][
            ~tgt_regression_padding
        ].flatten()
        tgt_watch_outside = (
            torch.cat(
                [t["gaze_watch_outside"][i] for t, (_, i) in zip(targets, indices)],
                dim=0,
            )[~tgt_regression_padding]
            .flatten()
            .long()
        )

        # If there are no targets, return
        if len(tgt_watch_outside) == 0 or len(pred_watch_outside) == 0:
            return

        self.metric(pred_watch_outside, tgt_watch_outside)
