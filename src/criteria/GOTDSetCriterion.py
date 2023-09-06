import torch
from torch import nn


class GOTDSetCriterion(nn.Module):
    def __init__(
        self,
        matcher: nn.Module,
        losses: dict[str, nn.Module],
    ):
        super().__init__()

        self.matcher = matcher
        self.losses = losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the preds of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        src_permutation_idx = self._get_src_permutation_idx(indices)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )

        # Compute all the requested losses
        losses = {}
        for loss_name, loss_fn in self.losses.items():
            loss = loss_fn(
                outputs_without_aux,
                targets,
                indices,
                src_permutation_idx=src_permutation_idx,
                num_boxes=num_boxes,
            )

            if loss is None:
                continue

            losses[loss_name] = loss

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)

                for loss_name, loss_fn in self.losses.items():
                    loss = loss_fn(
                        aux_outputs,
                        targets,
                        indices,
                        src_permutation_idx=src_permutation_idx,
                        num_boxes=num_boxes,
                    )

                    if loss is None:
                        continue

                    losses[f"{loss_name}_{i}"] = loss

        return losses
