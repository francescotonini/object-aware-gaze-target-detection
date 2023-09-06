import torch
from torchmetrics import Metric


class CounterMetric(Metric):
    def __init__(self):
        super().__init__()

        self.add_state(
            "counter",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(self, *args, **kwargs):
        self.counter += 1

    def compute(self):
        return self.counter
