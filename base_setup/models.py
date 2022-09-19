from torch import nn
from torch.nn import functional as F

class NeuralNet(nn.Module):
    def __init__(self, input_size=32, output_size=10) -> None:
        super().__init__()
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        return F.softmax(self.linear(x))
