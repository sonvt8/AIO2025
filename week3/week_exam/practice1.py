from localLib.sigmoid import SigmoidProcessor
import torch

x = torch.tensor([5.0, 3.0])
output = torch.sigmoid(x)
data = torch.Tensor([3.0, -2.0])
sig = SigmoidProcessor()
print(sig.sigTorch(data))
