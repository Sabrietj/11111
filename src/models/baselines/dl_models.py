import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    """
    CNN-LSTM 模型 (参考: Gohari et al., 2021)
    适用于从 all_flow.csv 提取出的序列化或统计流特征
    """

    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=2, dropout_rate=0.5):
        super(CNN_LSTM, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0, ceil_mode=True),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        x = self.conv1d_layers(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        logits = self.classifier(last_time_step_out)
        return logits


class DeepAMD(nn.Module):
    """
    DeepAMD 模型 (参考: Imtiaz et al., 2021)
    """

    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super(DeepAMD, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)