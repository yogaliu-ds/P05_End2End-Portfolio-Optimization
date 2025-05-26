import numpy as np
import torch
import torch.nn as nn

# End2End LSTM
class ScoreBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ScoreBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,  # now input_dim = num_assets
                            hidden_size=hidden_dim,
                            batch_first=True,
                            dropout=0.3,
                            bidirectional=False,
                            num_layers=2
                            )
        self.fc = nn.Linear(hidden_dim, input_dim)  # predict score for each asset

    def forward(self, x):
        # Input x: [B, N, T]
        x = x.permute(0, 2, 1)  # => [B, T, N]

        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)  # [B, T, H]
        last_output = lstm_out[:, -1, :]  # [B, H]

        scores = self.fc(last_output)  # [B, N]
        return scores
    
# End2End Single Linear Layer
# class ScoreBlock(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64):
#         super(ScoreBlock, self).__init__()
#         self.fc = nn.Linear(input_dim, input_dim)  # predict score for each asset

#     def forward(self, x):
#         # Input x: [B, N, T]
#         x = x[:, :, -1]
#         scores = self.fc(x)  # [B, N]
#         return scores