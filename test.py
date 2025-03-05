
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
from parse_config import get_model_parms, get_train_hparms
from dataset.dataset_processing import get_data_path
from models.resnet_3d_cnn import ResNet3DClassifier
from models.effb3_lstm import B3LSTMClassifier
from models.resnet50xt_lstm import ResNetLSTMClassifier
from sklearn.metrics import classification_report, roc_auc_score
import torchvision.transforms as transforms
from typing import Dict, Tuple, Optional, Any, List
from dataset.dataset import VideoDataset
from utils.same_augmentation import SameAugmentation
from utils.plot_metrics import plot_training_metrics, plot_confusion_matrix_final


def test_model(model: nn.Module, data_path: str, n_frames: int, size: Tuple[int, int]):
    batch_size = 1
    epoch_v_loss = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    transfor = SameAugmentation(
        transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )

    test_dataset = VideoDataset(n_frames, data_path, device, size, transfor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model.eval()
    
    all_val_labels, all_val_preds = [], []
    test_probs = []
    with torch.no_grad():
        for video_data, labels in tqdm(test_loader, desc="Test", leave=False):
            video_data, labels = video_data.to(device), labels.to(device)

            output = model(video_data)
            output_probs = torch.sigmoid(output)
            test_probs.extend(output_probs.cpu().numpy())

            loss = criterion(output, labels)
            epoch_v_loss += loss.item()

            output = torch.sigmoid(output).round()
            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(output.cpu().numpy())
        
    val_loss = epoch_v_loss / len(test_loader)
    val_accuracy = np.sum(np.array(all_val_labels) == np.array(all_val_preds)) / len(all_val_labels)

    print(f'Test loss: {val_loss:.4f}, Test accuracy: {val_accuracy:.4f}')

    print("Test Classification Report:")
    print(classification_report(all_val_labels, all_val_preds))
    plot_confusion_matrix_final(all_val_labels, all_val_preds, 'Test')
    val_auc = roc_auc_score(all_val_labels, test_probs)
    print(f'Test AUC: {val_auc}')


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a specified model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the model to train. E.g., 'MyModel'"
    )

    parser.add_argument(
        "--train_path",
        type=str,
        default="data/",
        help="Path to the dataset"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to the model checkpoint (required for test mode)."
    )

    args = parser.parse_args()
    _, _, test_path = get_data_path(args.train_path)

    if args.model.lower() == "resnet3d":
        parms = get_model_parms('hyperparameters_resnet3d')
        hparms: Dict[str, Any] = get_train_hparms('hyperparameters_resnet3d')
        model: nn.Module = ResNet3DClassifier(**parms)
    elif args.model.lower() == "effb3lstm":
        parms = get_model_parms('hyperparameters_eff_b3_lstm')
        hparms: Dict[str, Any] = get_train_hparms('hyperparameters_eff_b3_lstm')
        model: nn.Module = B3LSTMClassifier(**parms)
    elif args.model.lower() == "resnetlstm":
        parms = get_model_parms('hyperparameters_resnet_lstm')
        hparms: Dict[str, Any] = get_train_hparms('hyperparameters_resnet_lstm')
        model: nn.Module = ResNetLSTMClassifier(**parms)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_model(model, test_path, hparms['n_frames'], hparms['size'])


if __name__ == "__main__":
    main()