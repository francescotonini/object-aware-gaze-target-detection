import torch
from torch import nn


class GOTDEvaluation(nn.Module):
    def __init__(
        self,
        matcher: nn.Module,
        evals: dict,
    ):
        super().__init__()

        self.matcher = matcher
        self.evals = evals

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the preds of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        src_permutation_idx = self._get_src_permutation_idx(indices)

        for _, fn in self.evals.items():
            fn(
                outputs,
                targets,
                indices,
                src_permutation_idx=src_permutation_idx,
            )

    def reset(self):
        for _, fn in self.evals.items():
            fn.reset_metrics()

    def get_metrics(self):
        metrics = {}
        for eval_name, fn in self.evals.items():
            metrics.update(fn.get_metrics())

        return metrics
