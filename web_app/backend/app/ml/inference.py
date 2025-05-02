import torch
import torch.nn as nn
from app.ml.model import EffB3LSTMClassifier
from app.core.config import model_config


class VideoInference:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path: str) -> nn.Module:
        model = EffB3LSTMClassifier(
            lstm_hidden_dim=model_config.lstm_hidden_dim,
            num_lstm_layers=model_config.num_lstm_layers,
            dropout=model_config.dropout,
            bidirectional=model_config.bidirectional
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.to(self.device)
        return model

    def predict(self, frames: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(frames)
        return output
