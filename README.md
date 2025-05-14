# deepfake_video_detection_system
# Deepfake Video Detection System

This repository contains a comprehensive system for detecting deepfake videos using state-of-the-art machine learning models. It includes both a web application for user interaction and scripts for training and testing detection models.

## Project Overview

The Deepfake Video Detection System is designed to analyze video content and detect manipulations (deepfakes). It's a powerful tool for researchers and developers working in the domain of video authentication and security.

### Key Features:

- **Web Application:** A user-friendly interface for uploading and analyzing videos for deepfakes.
- **Model Training:** Scripts for training machine learning models on custom datasets.
- **Model Testing:** Easy-to-use scripts for evaluating the performance of trained models.

## How to Run the Web Application

### Prerequisites:

- **Install Docker:** Ensure Docker and Docker Compose are installed on your system. [Docker Installation Guide](https://docs.docker.com/get-docker/)
- **Environment Configuration:** Update all `.env` files in the `web_app` directory with the appropriate configuration values.

### Steps to Run:

1. Navigate to the web_app directory:
   ```bash
   cd web_app
   ```

2. Build and start the application using Docker Compose:
   ```bash
   docker compose up --build
   ```

3. Access the web application in your browser at `http://localhost:3000` (default port, adjust if configured differently in `.env` files).

## How to Train and Test Models

The repository includes scripts for training (`train.py`) and testing (`test.py`) models. These scripts utilize `argparse` for configuring runtime parameters.

### Prerequisites:

- **Prepare the Dataset:** Ensure your dataset is organized and accessible to the scripts.
- **Install Dependencies:** Install all required Python libraries:
  ```bash
  pip install -r requirements.txt
  ```

### Training a Model:

Execute the `train.py` script with the required arguments:

```bash
python train.py --data_path <path_to_dataset> --model_path <path_to_save_model> --epochs <number_of_epochs> --batch_size <batch_size>
```

Example:
```bash
python train.py --data_path ./data --model_path ./models/model.pth --epochs 20 --batch_size 32
```

### Testing a Model:

Execute the `test.py` script with the required arguments:

```bash
python test.py --data_path <path_to_test_data> --model_path <path_to_trained_model>
```

Example:
```bash
python test.py --data_path ./data --model_path ./models/model.pth
```

### Common Arguments:

| Argument | Description |
|----------|-------------|
| `--data_path` | Path to the dataset (training or testing data). |
| `--model_path` | Path to save or load the model. |
| `--epochs` | Number of training epochs (for `train.py`). |
| `--batch_size` | Batch size for training (for `train.py`). |

## Repository Structure

- `web_app/`: Contains the web application files and Docker configuration.
- `train.py`: Script for training models.
- `test.py`: Script for testing models.
- `data/`: Directory for storing datasets.
- `models/`: Directory for saving trained models.
- `requirements.txt`: File with Python dependencies.

## Notes

- **Environment Variables:** The `.env` files in the `web_app` directory must be correctly configured before starting the application.
- **Docker Configuration:** Ensure the Docker Compose file (`docker-compose.yml`) matches your system's requirements.
- **Model Training and Testing:** Familiarize yourself with the argparse options in `train.py` and `test.py` to customize the scripts for your specific use case.
