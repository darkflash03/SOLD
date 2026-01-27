import torch.nn as nn

class MLPDecoder(nn.Module):
    def __init__(
            self,
            n_classes: int = 4,
            encoder_hidden_dim: int = 640,
            mlp_num_layers: int = 3,
            mlp_dropout_p: float = 0.1,
            add_sigmoid: bool = False,
    ):
        super().__init__()
        if mlp_num_layers == 1:
            layers = [nn.Linear(encoder_hidden_dim, n_classes)]

        elif mlp_num_layers == 2:
            first_layer = [
                nn.Linear(encoder_hidden_dim, encoder_hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]
            final_layer = [
                nn.Linear(encoder_hidden_dim // 4, n_classes),
            ]
            layers = first_layer + final_layer

        else:
            assert mlp_num_layers >= 3
            num_hidden_layers = mlp_num_layers - 3

            first_layer = [
                nn.Linear(encoder_hidden_dim, encoder_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]

            second_layer = [
                nn.Linear(encoder_hidden_dim // 2, encoder_hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]

            hidden_layer = [
                nn.Linear(encoder_hidden_dim // 4, encoder_hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]

            final_layer = [
                nn.Linear(encoder_hidden_dim // 4, n_classes),
            ]

            layers = (
                    first_layer
                    + second_layer
                    + hidden_layer * num_hidden_layers
                    + final_layer
            )

        if add_sigmoid:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)