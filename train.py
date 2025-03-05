import os
import cv2
import json
import pandas as pd
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
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
from trainer.trainer import Trainer
from utils.plot_metrics import plot_training_metrics, plot_confusion_matrix_final


def main() -> None:
    """
    Main function to parse arguments, initialize the model and trainer, and start training.
    """
    parser = argparse.ArgumentParser(description="Train a specified model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the model to train. E.g., 'EffB3LSTM'"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/",
        help="Path to the dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run training on (default: cuda)"
    )
    args = parser.parse_args()
    train_path, val_path, test_path = get_data_path(args.train_path)

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


    early_stopping_params: Dict[str, Any] = {
        'patience': 15,
        'verbose': True,
        'delta': 0.0001,
        'path': 'custom_early_stopping_checkpoint.pt'
    }

    optimizer_params: Dict[str, Any] = {
        'lr': hparms['lr'],
        'weight_decay': hparms['weight_decay']
    }
    scheduler_params: Dict[str, Any] = {
        'milestones': hparms['milestones'],
        'gamma': hparms['gamma']
    }

    if args.optimaizer.lower() == 'SGD':
        optimaizer = optim.SGD(momentum=0.9)
    else:
        optimaizer = optim.Adam()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomRotation(degrees=5, fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainer: Trainer = Trainer(
        model, hparms, train_path, val_path, hparms['size'], torch.device(args.device),
        args.model, optimaizer, val_transform=val_transform, train_transform=train_transform,
        early_stopping_params=early_stopping_params, optimizer_params=optimizer_params, 
        scheduler_params=scheduler_params
    )
    
    trainer.train()


if __name__ == '__main__':
    main()