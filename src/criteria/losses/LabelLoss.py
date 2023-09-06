import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelLoss(nn.Module):
    def __init__(
        self,
        eos_coeff: int = 0.1,
        num_classes: int = 81,
        loss_weight: int = 1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.loss_weight = loss_weight

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coeff
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs, targets, indices, **kwargs):
        # If empty_weight is not on the same device as outputs["pred_logits"], set the device
        if self.empty_weight.device != outputs["pred_logits"].device:
            self.empty_weight = self.empty_weight.to(outputs["pred_logits"].device)

        idx = kwargs["src_permutation_idx"]

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o.argmax(-1)

        loss = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )

        return loss * self.loss_weight
