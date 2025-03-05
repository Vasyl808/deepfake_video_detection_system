import os
import random
import os.path
from typing import List, Dict, Tuple
import json

import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
import cv2
from utils.same_augmentation import SameAugmentation


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self,
        n_frames: int, df_path: str, device: nn.Module, 
        image_size : Tuple[int, int] = (224, 224), 
        transform : SameAugmentation = None
    ):
        self.n_frames = n_frames
        self.videos = []
        self.device = device if device is not None else torch.device("cpu")
        self.image_size = image_size

        self.transform = transform

        self.detector = MTCNN(device=device, post_process=False)

        with open(df_path) as f:
            videos = json.load(f)
            videos = [(video, metadata) for (video, metadata) in videos.items()]
            self.videos += videos

    def __getitem__(self, n: int) -> Tuple[torch.tensor, torch.FloatTensor]:
        video, metadata = self.videos[n]
        cap = cv2.VideoCapture(video)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = random.randint(0, max(0, total_frames - self.n_frames))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []

        for _ in range(self.n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resize = (self.image_size[0], self.image_size[1])
            frame = cv2.resize(frame, resize)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)

        cap.release()

        frames = torch.stack(frames)

        if self.transform:
            frames = self.transform(frames)

        label = 0.0
        if metadata['label'] == 'FAKE':
            label = 1.0

        return frames, torch.FloatTensor([label])


    def __len__(self):
        return len(self.videos)

