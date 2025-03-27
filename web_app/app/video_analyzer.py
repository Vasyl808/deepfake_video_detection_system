import cv2 
import torch
import numpy as np
import base64
import matplotlib.pyplot as plt
from torchvision import transforms
from schemas import FrameResult, AnalysisResponse, SequenceAnalysis
from utils import extract_frames, find_last_conv_layer, GradCAM, SameAugmentation
from ml_model import EffB3LSTMClassifier


class VideoAnalyzer:
    def __init__(self):
        self.device = 'cpu'
        self.model = self._load_model()
    
    def _load_model(self):
        hyperparameters = {
            'n_linear_hidden': 128,
            'lstm_hidden_dim': 32,
            'num_lstm_layers': 2,
            'dropout': 0.4,
            'bidirectional': False,
            'freeze': False,
            'n_frames': 20,
        }


        model = EffB3LSTMClassifier(
            n_linear_hidden=hyperparameters['n_linear_hidden'],
            lstm_hidden_dim=hyperparameters['lstm_hidden_dim'],
            num_lstm_layers=hyperparameters['num_lstm_layers'],
            dropout=hyperparameters['dropout'],
            bidirectional=hyperparameters['bidirectional'],
            freeze=hyperparameters['freeze']
        )

        model.load_state_dict(torch.load('best_ResNetLSTMNoBiClassifier20x224_lstm_checkpoint_epoch_step_23.pt', map_location=torch.device('cpu'))['model_state_dict'])
        model.to(self.device)
        return model
    
    def _analyze_sequence(self, sequence) -> SequenceAnalysis:
        num_frames = len(sequence)
        #print(f"Обрана послідовність містить {num_frames} кадр(ів).")
        
        transfor = SameAugmentation(
            transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        )
        frames = torch.stack([frame for frame in sequence])  # (num_frames, channels, height, width)
        frames = transfor(frames).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            output = self.model(frames)
            classification = output.cpu().numpy().tolist()[0]
 
        target_module = self.model.feature_extractor[0]  
        target_layer = find_last_conv_layer(target_module)

        gradcam = GradCAM(self.model, target_layer, input_size=(224, 224))

        frames.requires_grad = True
        grad_cam_maps = gradcam.generate(frames)

        output = self.model(frames)
        num_frames = grad_cam_maps.shape[0]

        sequence_results = []
        grad_cam_results = []

        video_tensor_cpu = frames.cpu().squeeze(0)  # [num_frames, 3, 224, 224]
        for idx in range(num_frames):
            frame_tensor = video_tensor_cpu[idx]
            frame_np = frame_tensor.permute(1, 2, 0).detach().numpy()  # (224,224,3)
            frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min() + 1e-8)
            frame_np = np.uint8(frame_np * 255)
            frame_np_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', frame_np_bgr)
            img_str = base64.b64encode(buffer).decode("utf-8")
            sequence_results.append(FrameResult(frame_number=idx, image=img_str))
        
        for i in range(num_frames):
            frame_np = video_tensor_cpu[i].permute(1, 2, 0).detach().numpy()
            frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min() + 1e-8)
            frame_np = np.uint8(255 * frame_np)
                
            heatmap = np.uint8(255 * grad_cam_maps[i])
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(frame_np, 0.5, heatmap, 0.5, 0)

            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', overlay)
            gradcam_img_str = base64.b64encode(buffer).decode("utf-8")
            grad_cam_results.append(FrameResult(frame_number=i, image=gradcam_img_str))

        is_fake = bool(torch.sigmoid(torch.tensor(classification[0])).round())
        explanation = (
            "Послідовність визначено як фейк."
            if is_fake 
            else "Послідовність виглядає реальним."
        )
        
        return SequenceAnalysis(
            classification=[bool(torch.sigmoid(torch.tensor(classification[0])).round())],
            is_fake=is_fake,
            explanation=explanation,
            frames=sequence_results,
            gradcam=grad_cam_results
        )
    
    def analyze(self, video_path: str, start_time: int, duration: int) -> AnalysisResponse:
        end_time = start_time + duration
        frames = extract_frames(video_path, start_time, end_time, num_frames=20)
        if not frames:
            raise ValueError("Не вдалося отримати кадри з відео")
        sequences_results = []
        for sequence in frames:
            result = self._analyze_sequence(sequence)
            sequences_results.append(result)
        
        return AnalysisResponse(
            sequences=sequences_results
        )
