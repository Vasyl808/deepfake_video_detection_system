import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from facenet_pytorch import MTCNN
from torchvision import transforms


class SameAugmentation:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, frames):
        seed = torch.randint(0, 2**32, (1,)).item()

        augmented_frames = []
        for frame in frames:
            torch.manual_seed(seed)
            augmented_frames.append(self.augmentations(frame))
        
        return torch.stack(augmented_frames)


class GradCAM:
    def __init__(self, model, target_layer, input_size=(224, 224)):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.input_size = input_size
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_logit=None):

        output = self.model(input_tensor)  # (1, 1)
        self.model.zero_grad()

        output.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (num_frames, C, 1, 1)

        grad_cam_map = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (num_frames, 1, H_feat, W_feat)

        grad_cam_map = F.interpolate(grad_cam_map, size=self.input_size, mode='bilinear', align_corners=False)
        grad_cam_map = grad_cam_map.squeeze(1)

        grad_cam_maps = []
        for i in range(grad_cam_map.size(0)):
            cam = grad_cam_map[i]
            cam_min = cam.min()
            cam_max = cam.max()
            cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            grad_cam_maps.append(cam_norm.cpu().numpy())
        return np.stack(grad_cam_maps, axis=0)


def find_last_conv_layer(module):
    last_conv = None
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d):
            last_conv = layer
    return last_conv


def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector = MTCNN(device='cpu', post_process=False)
    boxes, _ = detector.detect(frame, landmarks=False)

    if boxes is None:
        return None

    faces = []
    for box in boxes:
        width = box[2] - box[0]
        height = box[3] - box[1]
        expand_x = width * 0.3 / 2
        expand_y = height * 0.3 / 2
        x1 = max(int(box[0] - expand_x), 0)
        y1 = max(int(box[1] - expand_y), 0)
        x2 = min(int(box[2] + expand_x), frame.shape[1])
        y2 = min(int(box[3] + expand_y), frame.shape[0])

        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (224, 224))
        face = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
        faces.append((face, box))

    return faces


def process_roi(frame, roi):
    x1, y1, x2, y2 = roi
    # Забезпечуємо, що координати - це цілі значення
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Вирізаємо область
    cropped = frame[y1:y2, x1:x2]
    processed_faces = process_frame(cropped)
    # Коригуємо координати виявлених облич відносно повного кадру
    adjusted_faces = []
    if processed_faces is not None:
        for face, box in processed_faces:
            adjusted_box = (box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1)
            adjusted_faces.append((face, adjusted_box))
    return adjusted_faces


def extract_frames(video_path, start_time, end_time, num_frames=20, margin=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_sec = total_frames / fps

    if end_time > video_duration_sec:
        end_time = video_duration_sec

    segment_duration = end_time - start_time
    if segment_duration <= 0:
        cap.release()
        return []

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frames_to_consider = end_frame - start_frame

    frame_indices = np.linspace(0, frames_to_consider - 1, num_frames, dtype=int)

    face_sequences = {}
    # Обробка першого кадру: повний пошук облич
    first_frame_index = start_frame + frame_indices[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_index)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []
    
    processed_faces = process_frame(frame)
    if processed_faces is not None:
        for face, box in processed_faces:
            face_sequences[len(face_sequences)] = {
                "roi": box,
                "faces": [face]
            }
    # Обробка наступних кадрів з використанням ROI
    for idx in frame_indices[1:]:
        frame_index = start_frame + idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            continue

        for seq in face_sequences.values():
            x1, y1, x2, y2 = seq["roi"]
            height, width = frame.shape[:2]
            new_x1 = max(int(x1) - margin, 0)
            new_y1 = max(int(y1) - margin, 0)
            new_x2 = min(int(x2) + margin, width)
            new_y2 = min(int(y2) + margin, height)
            roi = (new_x1, new_y1, new_x2, new_y2)

            detected_faces = process_roi(frame, roi)
            if detected_faces:
                face, new_box = detected_faces[0]
                seq["faces"].append(face)
                seq["roi"] = new_box
            else:
                seq["faces"].append(seq["faces"][-1])

    cap.release()
    for seq in face_sequences.values():
        if len(seq["faces"]) < num_frames:
            seq["faces"].extend([seq["faces"][-1]] * (num_frames - len(seq["faces"])))

    return [seq["faces"] for seq in face_sequences.values()]