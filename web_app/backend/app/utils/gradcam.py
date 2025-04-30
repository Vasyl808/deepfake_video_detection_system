import torch
import numpy as np
from typing import Tuple
import torch.nn as nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: torch.nn.Conv2d, input_size: Tuple[int] = (224, 224)) -> None:
        self.model = model
        self.target_layer = target_layer
        self.input_size = input_size
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module: nn.Module, input: torch.tensor, output: torch.tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_in: torch.tensor, grad_out: torch.tensor) -> None:
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.hook_handles.append(
            self.target_layer.register_backward_hook(backward_hook)
        )

    def generate(self, input_tensor: torch.tensor) -> np.stack:
        # Forward
        output = self.model(input_tensor)

        # Assume binary classification
        class_idx = output.argmax(dim=1).item()
        score = output[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        grad_cam = (weights * self.activations).sum(dim=1, keepdim=True)
        grad_cam = torch.relu(grad_cam)
        grad_cam = torch.nn.functional.interpolate(
            grad_cam, self.input_size, mode='bilinear', align_corners=False
        )
        grad_cam = grad_cam.squeeze().cpu().numpy()

        # Normalize for each frame
        if len(grad_cam.shape) == 2:
            grad_cam = np.expand_dims(grad_cam, 0)
        grad_cam_norm = []
        for m in grad_cam:
            m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)
            grad_cam_norm.append(m_norm)
        grad_cam_norm = np.stack(grad_cam_norm)
        return grad_cam_norm

    def remove_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()


def find_last_conv_layer(module: nn.Module) -> nn.Conv2d:
    conv = None
    
    for child in module.children():
        if list(child.children()):
            sub_conv = find_last_conv_layer(child)
            if sub_conv:
                conv = sub_conv
        elif isinstance(child, torch.nn.Conv2d):
            conv = child
            
    return conv