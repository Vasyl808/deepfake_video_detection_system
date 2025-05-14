import cv2
import numpy as np
import base64
import torch
import cv2
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from facenet_pytorch import MTCNN
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


def tensor_to_b64(frame_tensor):
    frame_np = frame_tensor.permute(1, 2, 0).detach().numpy()
    frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min() + 1e-8)
    frame_np = np.uint8(frame_np * 255)
    frame_np_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', frame_np_bgr)
    return base64.b64encode(buffer).decode("utf-8")


def overlay_heatmap(tensor, gradcam_map, out_path):
    frame_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
    frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min() + 1e-8)
    frame_np = np.uint8(255 * frame_np)

    heatmap = np.uint8(255 * gradcam_map)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(frame_np, 0.5, heatmap, 0.5, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(out_path, overlay)


def tensor_to_imagefile(tensor, out_path):
    frame_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
    frame_np = (frame_np - frame_np.min()) / (frame_np.max() - frame_np.min() + 1e-8)
    frame_np = np.uint8(255 * frame_np)
    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(out_path, frame_np)


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
    