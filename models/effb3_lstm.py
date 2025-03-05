import torch.nn as nn
import torch
from torchvision import models
from base_model import BaseModel


class B3LSTMClassifier(BaseModel):
    def __init__(
        self, lstm_hidden_dim: int, num_lstm_layers: int,
        dropout: float, bidirectional: bool
    ):
        super(B3LSTMClassifier, self).__init__()

        self.cnn = models.efficientnet_b3(pretrained=True)
        self.feature_output_size = 1536

        self.feature_extractor = nn.Sequential(
            *list(self.cnn.children())[:-1],
        )

        self.lstm = nn.LSTM(
            input_size=self.feature_output_size,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=0.1,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim // 2, 1),
        )

    def forward(self, vid_frames: torch.tensor) -> torch.tensor:
        batch_size, num_frames, channels, height, width = vid_frames.shape
        vid_frames = vid_frames.view(batch_size * num_frames, channels, height, width)

        vid_features = self.feature_extractor(vid_frames)  # (batch_size*num_frames, feature_output_size, 1, 1)
        vid_features = vid_features.view(batch_size, num_frames, -1)  # (batch_size, num_frames, feature_output_size)

        lstm_out, _ = self.lstm(vid_features)  # (batch_size, num_frames, lstm_hidden_dim * num_directions)

        output = self.classifier(lstm_out[:, -1, :])  # (batch_size, 1)

        return output
