import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import List, Optional

load_dotenv()


class FastAPIConfig(BaseSettings):
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = os.getenv("PORT", 8000)

    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp_videos")
    RESULTS_DIR: str = os.environ.get("RESULTS_DIR", "./analyzed_frames")
    MAX_FILE_SIZE: int = os.getenv("MAX_FILE_SIZE", 500 * 1024 * 1024)

    secret_key: str = os.getenv("SECRET_KEY")

    FILE_TTL: int = 3600
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")


class ModelConfig(BaseSettings):
    n_frames: int = os.getenv("N_FRAMES", 20)
    lstm_hidden_dim: int = os.getenv("LSTM_HIDDEN_DIM", 32)
    num_lstm_layers: int = os.getenv("NUM_LSTM_LAYERS", 2)
    bidirectional: bool = os.getenv("BIDIRECTIONAL", False)
    dropout: float = os.getenv("DROPOUT", 0.4)
    freeze: bool = os.getenv("FREEZE", False)
    mean: List[float] = os.getenv("MEAN", [0.485, 0.456, 0.406])
    std: List[float] = os.getenv("STD", [0.229, 0.224, 0.225])
    img_size: int = os.getenv("IMG_SIZE", 224)
    device: str = os.getenv("DEVICE", "cpu")
    file_path: str = os.getenv("MODEL_FILE_PATH", "app/ml/weights/best_ResNetLSTMNoBiClassifier20x224_lstm_checkpoint_epoch_step_23.pt")


app_config = FastAPIConfig()
model_config = ModelConfig()