import torch
from torchvision import transforms
from app.ml.inference import VideoInference
from app.schemas.analysis import AnalysisResponse, SequenceAnalysis
from app.schemas.frames import FrameResult
from app.utils.gradcam import GradCAM, find_last_conv_layer
from app.utils.frames import tensor_to_b64, overlay_heatmap, extract_frames
from app.utils.augmentation import SameAugmentation
from app.core.config import model_config


class VideoAnalyzer:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.inference = VideoInference(model_path, device)

    def _analyze_sequence(self, sequence) -> SequenceAnalysis:
        transform = SameAugmentation(
            transforms.Compose([
                transforms.Normalize(mean=model_config.mean, std=model_config.std),
            ])
        )
        frames = torch.stack([frame for frame in sequence])
        frames = transform(frames).unsqueeze(0)

        output = self.inference.predict(frames)
        score = output.cpu().numpy().tolist()[0]

        target_module = self.inference.model.feature_extractor[0]
        target_layer = find_last_conv_layer(target_module)
        gradcam = GradCAM(self.inference.model, target_layer, input_size=(224, 224))

        frames.requires_grad = True
        grad_cam_maps = gradcam.generate(frames)
        num_frames = grad_cam_maps.shape[0]

        video_tensor_cpu = frames.cpu().squeeze(0)
        sequence_results = [
            FrameResult(frame_number=i, image=tensor_to_b64(video_tensor_cpu[i]))
            for i in range(num_frames)
        ]

        grad_cam_results = [
            FrameResult(frame_number=i, image=overlay_heatmap(video_tensor_cpu[i], grad_cam_maps[i]))
            for i in range(num_frames)
        ]

        is_fake = bool(torch.sigmoid(torch.tensor(score[0])).round())
        explanation = (
            "Послідовність визначено як фейк."
            if is_fake
            else "Послідовність облич виглядає реальною."
        )

        return SequenceAnalysis(
            classification=[is_fake],
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

        sequences_results = [self._analyze_sequence(seq) for seq in frames]

        return AnalysisResponse(sequences=sequences_results)


video_analyzer = VideoAnalyzer(model_config.file_path, model_config.device)