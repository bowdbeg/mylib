import numpy as np
import torch
from pathlib import Path
import dataclasses


@dataclasses.dataclass
class EarlyStopper:
    patience: int
    delta: float = 0.0
    path: Path = Path("checkpoint.pt")

    def __post_init__(self) -> None:
        self.coutner = 0
        self.best_score = -float("inf")
        self.early_stop = False
        self.patience = self.patience if self.patience else float("inf")

    def __call__(self, score, model) -> bool:

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
