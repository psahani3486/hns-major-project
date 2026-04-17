import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import functional as TF


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> torch.Tensor:
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

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


def denormalize_image_tensor(x: torch.Tensor, mean, std) -> np.ndarray:
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


def load_bundle(bundle_path: Path):
    bundle = torch.load(bundle_path, map_location="cpu")
    return bundle


def main():
    parser = argparse.ArgumentParser(description="Load a saved lung disease model bundle and generate a saliency map for one image")
    parser.add_argument("--bundle", required=True, help="Path to a .pt bundle saved after training")
    parser.add_argument("--image", required=True, help="Path to a chest X-ray image")
    parser.add_argument("--output-dir", default="saliency_output", help="Directory to save the heatmap and overlay")
    parser.add_argument("--class-idx", type=int, default=None, help="Optional target class index; if omitted, uses predicted class")
    args = parser.parse_args()

    bundle = load_bundle(Path(args.bundle))
    backbone = bundle["backbone"]
    pretrained = bool(bundle["pretrained"])
    class_names = bundle["class_names"]
    image_size = int(bundle["image_size"])
    mean = bundle.get("mean", [0.485, 0.456, 0.406])
    std = bundle.get("std", [0.229, 0.224, 0.225])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone, num_classes=len(class_names), pretrained=pretrained)
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    image = Image.open(args.image).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    target_layer = get_target_layer(model, backbone)
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate(x, class_idx=args.class_idx)
    logits = model(x)
    predicted_idx = int(torch.argmax(logits, dim=1).item())
    predicted_label = class_names[predicted_idx]
    grad_cam.remove()

    rgb_img = denormalize_image_tensor(x, mean, std)
    heat_rgb = heatmap_to_rgb(heatmap.numpy())
    overlay = np.clip(0.6 * rgb_img + 0.4 * heat_rgb, 0.0, 1.0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    Image.fromarray((heat_rgb * 255).astype(np.uint8)).save(out_dir / f"{stem}_heatmap.png")
    Image.fromarray((overlay * 255).astype(np.uint8)).save(out_dir / f"{stem}_overlay.png")
    Image.fromarray((rgb_img * 255).astype(np.uint8)).save(out_dir / f"{stem}_input.png")

    print(f"Predicted class: {predicted_label}")
    print(f"Saved input, heatmap, and overlay to: {out_dir}")


if __name__ == "__main__":
    main()
