import torch


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
        