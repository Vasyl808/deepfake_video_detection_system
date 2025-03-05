import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score
import torchvision.transforms as transforms
from typing import Dict, Tuple, Optional, Any, List

from utils.early_stopping import EarlyStopping
from dataset.dataset import VideoDataset
from utils.same_augmentation import SameAugmentation
from utils.plot_metrics import plot_training_metrics, plot_confusion_matrix_final


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        hparms: Dict[str, Any],
        path_train: str,
        path_test: str,
        size: Tuple[int, int],
        device: torch.device,
        name: str,
        optimaizer: nn.Module,
        checkpoint_path: Optional[str] = None,
        early_stopping_params: Optional[Dict[str, Any]] = None,
        criterion: Optional[nn.Module] = None,
        optimizer_params: Optional[Dict[str, Any]] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        train_transform: Optional[transforms.Compose] = None,
        val_transform: Optional[transforms.Compose] = None
    ) -> None:
        """
        Initialize the Trainer.

        Parameters:
            model: The model to train.
            hparms: Hyperparameters including batch_size, num_epochs, n_frames, lr, gamma, milestones.
            path_train: Path to training data.
            path_test: Path to validation/test data.
            size: Expected size (e.g., image dimensions) for the dataset.
            device: Device to run training on (e.g., 'cpu' or 'cuda').
            name: Name used for saving checkpoints.
            checkpoint_path: Optional pre-existing checkpoint path.
            early_stopping_params: Dict of parameters for EarlyStopping.
            criterion: Loss function. If None, defaults to nn.BCEWithLogitsLoss().
            optimizer_params: Dict with optimizer configuration. Expected keys: lr and weight_decay.
            scheduler_params: Dict with scheduler configuration. Expected keys: milestones and gamma.
            train_transform: Transformations for training data. If None, default augmentation is used.
            val_transform: Transformations for validation data. If None, default normalization is used.
        """
        self.model: nn.Module = model
        self.hparms: Dict[str, Any] = hparms
        self.path_train: str = path_train
        self.path_test: str = path_test
        self.size: Tuple[int, int] = size
        self.device: torch.device = device
        self.name: str = name
        self.checkpoint_path: Optional[str] = checkpoint_path

        self.batch_size: int = hparms.get('batch_size', 4)
        self.num_epochs: int = hparms.get('num_epochs', 25)
        self.n_frames: int = hparms.get('n_frames', 20)
        self.lr: float = hparms.get('lr', 0.00001)
        self.gamma: float = hparms.get('gamma', 0.1)
        self.milestones: List[int] = hparms.get('milestones', [32, 64])

        self.model.to(self.device)

        if early_stopping_params is None:
            early_stopping_params = {
                'patience': 15,
                'verbose': True,
                'delta': 0.0001,
                'path': f'early_stopping_{self.name}_checkpoint.pt'
            }

        self.early_stopping = EarlyStopping(**early_stopping_params)

        self.criterion: nn.Module = criterion if criterion is not None else nn.BCEWithLogitsLoss()
  
        if optimizer_params is None:
            optimizer_params = {'lr': self.lr, 'weight_decay': 5e-4}
        self.optimizer = optimaizer(self.model.parameters(), **optimizer_params)
        
        if scheduler_params is None:
            scheduler_params = {'milestones': self.milestones, 'gamma': self.gamma}
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, **scheduler_params)
        
        if train_transform is None:
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
        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        self.train_transform = SameAugmentation(train_transform)
        self.val_transform = SameAugmentation(val_transform)
        
        self.train_dataset = VideoDataset(
            self.n_frames, self.path_train, self.device, self.size,
            self.train_transform
        )
        self.val_dataset = VideoDataset(
            self.n_frames, self.path_test, self.device, self.size,
            self.val_transform
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def load_checkpoint(self) -> int:
        if self.checkpoint_path:
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint: Dict[str, Any] = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch: int = checkpoint['epoch']
            print(f"Resuming training at epoch {start_epoch}")
            return start_epoch
        return 0

    def train_one_epoch(self, epoch: int) -> Tuple[float, float, List[Any], List[Any]]:
        epoch_loss: float = 0.0
        wrong_preds: int = 0
        total_samples: int = 0
        all_labels: List[Any] = []
        all_preds: List[Any] = []
        
        self.model.train()
        for video_data, labels in tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs} Training",
            leave=False
        ):
            video_data, labels = video_data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(video_data)
            loss = self.criterion(outputs, labels)
            epoch_loss += loss.item()
            
            outputs_sigmoid = torch.sigmoid(outputs)
            preds = outputs_sigmoid.round()

            wrong_preds += (labels - preds).abs().sum().item()
            total_samples += labels.shape[0]
            
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            
            loss.backward()
            self.optimizer.step()
        avg_loss: float = epoch_loss / len(self.train_loader)
        accuracy: float = (total_samples - wrong_preds) / total_samples
        return avg_loss, accuracy, all_labels, all_preds

    def evaluate(self) -> Tuple[float, float, float, List[Any], List[Any]]:
        self.model.eval()
        epoch_loss: float = 0.0
        all_labels: List[Any] = []
        all_preds: List[Any] = []
        all_probs: List[Any] = []
        with torch.no_grad():
            for video_data, labels in tqdm(
                self.val_loader,
                desc="Validation",
                leave=False
            ):
                video_data, labels = video_data.to(self.device), labels.to(self.device)
                outputs = self.model(video_data)
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()
                
                outputs_sigmoid = torch.sigmoid(outputs)
                all_probs.extend(outputs_sigmoid.cpu().numpy())
                preds = outputs_sigmoid.round()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        avg_loss: float = epoch_loss / len(self.val_loader)
        accuracy: float = np.sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)
        auc: float = roc_auc_score(all_labels, all_probs)
        return avg_loss, accuracy, auc, all_labels, all_preds

    def save_checkpoint(self, epoch: int) -> None:
        checkpoint_file: str = f'best_{self.name}_lstm_checkpoint_epoch_{epoch + 1}.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch + 1
        }, checkpoint_file)
        print(f"Checkpoint saved: {checkpoint_file}")

    def train(self) -> None:
        start_time: datetime.datetime = datetime.datetime.now()
        print(f"Training started at {start_time}, using device: {self.device}")
        
        train_losses: List[float] = []
        train_accuracies: List[float] = []
        val_losses: List[float] = []
        val_accuracies: List[float] = []
        epoch_durations: List[float] = []
        overall_train_labels: List[Any] = []
        overall_train_preds: List[Any] = []
        
        start_epoch: int = self.load_checkpoint()
        
        for epoch in range(start_epoch, self.num_epochs):
            epoch_start: float = time.time()
            
            # Training phase
            train_loss, train_acc, epoch_train_labels, epoch_train_preds = self.train_one_epoch(epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            overall_train_labels.extend(epoch_train_labels)
            overall_train_preds.extend(epoch_train_preds)
            
            # Evaluation phase
            val_loss, val_acc, val_auc, val_labels, val_preds = self.evaluate()
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            epoch_duration: float = time.time() - epoch_start
            epoch_durations.append(epoch_duration)
            
            print(f"Epoch {epoch + 1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
            print(f"  Duration: {epoch_duration:.2f} sec")
            
            self.scheduler.step()

            self.save_checkpoint(epoch)
            
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered. Terminating training loop.")
                break
        
        total_training_time: datetime.timedelta = datetime.datetime.now() - start_time
        print(f"Training completed in: {total_training_time}")

        plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
        
        print("Train Classification Report:")
        print(classification_report(overall_train_labels, overall_train_preds))
        plot_confusion_matrix_final(overall_train_labels, overall_train_preds, 'Train')
        
        print("Validation Classification Report:")
        print(classification_report(val_labels, val_preds))
        plot_confusion_matrix_final(val_labels, val_preds, 'Validation')
        print(f"Final Validation AUC: {val_auc:.4f}")