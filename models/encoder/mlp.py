import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 640,
        hidden_dims: list = [64],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)