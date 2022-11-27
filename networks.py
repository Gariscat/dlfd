import torch
import torch.nn as nn

class RNNPredictor(nn.Module):
    """
    :param obs_ord: maximum order of the input
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self,
        obs_ord: int = 3,
        hidden_size: int = 32,
        num_layers: int = 3,
        rnn: nn.Module = nn.LSTM
    ):
        super().__init__()

        self.rnn = rnn(
            input_size=obs_ord,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            # nn.Flatten(),
        )

    def forward(self, observations):
        # print(type(observations), observations.shape, type(self.net))
        last_hidden_states, (_, _) = self.rnn(observations)
        # print(last_hidden_states.shape)
        t = last_hidden_states[:, -1, :]
        preds = self.mlp(t)
        return preds
