import math
import torch


class SigmoidProcessor:
    def sigmoid(self, val: float) -> float:
        return 1 / (1 + math.exp(-val))

    def sigList(self, aList: list) -> list:
        return [self.sigmoid(i) for i in aList]

    def sigTorch(self, aTorch: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(aTorch)
