import torch
from torchvision import transforms
from typing import List


class SameAugmentation:
    def __init__(self, augmentations: List[transforms]):
        self.augmentations = augmentations

    def __call__(self, frames: torch.tensor) -> torch.tensor:
        seed = torch.randint(0, 2**32, (1,)).item()

        augmented_frames = []
        for frame in frames:
            torch.manual_seed(seed)
            augmented_frames.append(self.augmentations(frame))

        return torch.stack(augmented_frames)
        