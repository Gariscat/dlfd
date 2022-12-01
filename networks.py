import torch
import torch.nn as nn
from scipy.signal import stft
import numpy as np


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
        rnn: nn.Module = nn.LSTM,
        seq_len: int = 256
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


class RNNPredictorSpecAug(nn.Module):
    """
    :param obs_ord: maximum order of the input
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self,
        obs_ord: int = 3,
        hidden_size: int = 32,
        num_layers: int = 3,
        rnn: nn.Module = nn.LSTM,
        seq_len: int = 256,
    ):
        super().__init__()
        # n_fft = seq_len
        input_size = obs_ord + 2 * (1+seq_len//2)
        self.rnn = rnn(
            input_size=input_size,
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
        raw = observations[:, :, 0].flatten().cpu().numpy()  # (batch_size, seq_len, dim)
        seq_len = raw.shape[0]
        _, __, spec = stft(raw, nperseg=seq_len)
        mag_0, pha_0 = np.abs(spec)[:, 0], np.angle(spec)[:, 0]
        aug = torch.cat((torch.from_numpy(mag_0), torch.from_numpy(pha_0)))
        aug = torch.stack([aug]*seq_len).unsqueeze(0)
        aug = aug.to(observations.device)
        observations = torch.cat((observations, aug), dim=-1)

        # print(type(observations), observations.shape, type(self.net))
        last_hidden_states, (_, _) = self.rnn(observations)
        # print(last_hidden_states.shape)
        t = last_hidden_states[:, -1, :]
        preds = self.mlp(t)
        return preds