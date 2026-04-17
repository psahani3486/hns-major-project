from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


@dataclass
class LoadedBundle:
    model: nn.Module
    backbone: str
    class_names: list[str]
    image_size: int
    mean: list[float]
    std: list[float]


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> torch.Tensor:
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        target = logits[:, class_idx]
        self.model.zero_grad(set_to_none=True)
        target.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)

        cam_min, cam_max = cam.min(), cam.max()
        if float(cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu()


def build_model(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
    if backbone == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def get_target_layer(model: nn.Module, backbone: str) -> nn.Module:
    if backbone == "alexnet":
        return model.features[-1]
    if backbone == "resnet18":
        return model.layer4[-1].conv2
    if backbone == "densenet121":
        return model.features.denseblock4.denselayer16.conv2
    if backbone == "efficientnet_b0":
        return model.features[-1][0]
    raise ValueError(f"Unsupported backbone: {backbone}")


def denormalize_image_tensor(x: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    mean_t = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    img = x * std_t + mean_t
    img = torch.clamp(img, 0.0, 1.0)
    return img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    h = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 * h - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * h - 1.0) * 1.5, 0.0, 1.0)
    b = np.clip(1.0 - 1.5 * h, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def find_default_bundle_path(project_root: Path) -> Path:
    candidates = sorted(project_root.glob("outputs_paper_seed3_ep10/**/*_deployment.pt"))
    if not candidates:
        raise FileNotFoundError("No deployment bundle found under outputs_paper_seed3_ep10")
    return candidates[0]


def load_bundle(bundle_path: Path, device: torch.device) -> LoadedBundle:
    raw = torch.load(bundle_path, map_location="cpu")
    backbone = str(raw["backbone"])
    class_names = list(raw["class_names"])
    image_size = int(raw["image_size"])
    mean = list(raw.get("mean", [0.485, 0.456, 0.406]))
    std = list(raw.get("std", [0.229, 0.224, 0.225]))
    model = build_model(backbone=backbone, num_classes=len(class_names), pretrained=bool(raw["pretrained"]))
    model.load_state_dict(raw["model_state_dict"])
    model.to(device)
    model.eval()

    return LoadedBundle(
        model=model,
        backbone=backbone,
        class_names=class_names,
        image_size=image_size,
        mean=mean,
        std=std,
    )


def image_bytes_to_tensor(image_bytes: bytes, image_size: int, mean: list[float], std: list[float], device: torch.device) -> torch.Tensor:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    t = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return t(image).unsqueeze(0).to(device)


def to_png_base64(arr: np.ndarray) -> str:
    image = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def rank_probabilities(probs: np.ndarray, class_names: list[str]) -> list[dict[str, float | str]]:
    ranked = sorted(
        [
            {
                "class_name": class_names[i],
                "probability": float(probs[i]),
            }
            for i in range(len(class_names))
        ],
        key=lambda r: float(r["probability"]),
        reverse=True,
    )
    return ranked


def summarize_probabilities(probs: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    safe_probs = np.clip(probs.astype(float), 1e-12, 1.0)
    pred_idx = int(np.argmax(safe_probs))
    sorted_idx = np.argsort(safe_probs)[::-1]
    top2_idx = sorted_idx[:2].tolist()

    entropy = float(-(safe_probs * np.log(safe_probs)).sum())
    max_entropy = float(np.log(len(class_names))) if class_names else 1.0
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    margin = float(safe_probs[top2_idx[0]] - safe_probs[top2_idx[1]]) if len(top2_idx) > 1 else float(safe_probs[top2_idx[0]])

    return {
        "predicted_index": pred_idx,
        "predicted_label": class_names[pred_idx],
        "top_confidence": float(safe_probs[pred_idx]),
        "confidence_margin_top1_top2": margin,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "top2": [
            {
                "class_name": class_names[i],
                "probability": float(safe_probs[i]),
            }
            for i in top2_idx
        ],
    }


def predict_probabilities(image_bytes: bytes, loaded: LoadedBundle, device: torch.device) -> np.ndarray:
    x = image_bytes_to_tensor(image_bytes, loaded.image_size, loaded.mean, loaded.std, device)
    with torch.no_grad():
        logits = loaded.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(float)
    return probs


def predict_with_explanation(image_bytes: bytes, loaded: LoadedBundle, device: torch.device, class_idx: int | None = None) -> dict[str, Any]:
    x = image_bytes_to_tensor(image_bytes, loaded.image_size, loaded.mean, loaded.std, device)

    with torch.no_grad():
        logits = loaded.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(float)
        pred_idx = int(np.argmax(probs))

    target_layer = get_target_layer(loaded.model, loaded.backbone)
    grad_cam = GradCAM(loaded.model, target_layer)
    cam = grad_cam.generate(x, class_idx=class_idx if class_idx is not None else pred_idx)
    grad_cam.remove()

    input_rgb = denormalize_image_tensor(x, loaded.mean, loaded.std)
    heat_rgb = heatmap_to_rgb(cam.numpy())
    overlay = np.clip(0.6 * input_rgb + 0.4 * heat_rgb, 0.0, 1.0)

    ranked = rank_probabilities(probs, loaded.class_names)
    analysis = summarize_probabilities(probs, loaded.class_names)

    return {
        "predicted_index": pred_idx,
        "predicted_label": loaded.class_names[pred_idx],
        "class_probabilities": ranked,
        "analysis": analysis,
        "images": {
            "input_png_base64": to_png_base64(input_rgb),
            "heatmap_png_base64": to_png_base64(heat_rgb),
            "overlay_png_base64": to_png_base64(overlay),
        },
    }
