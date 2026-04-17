import argparse
import itertools
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as TF


@dataclass
class ExperimentConfig:
    backbone: str
    pretrained: bool
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    image_size: int
    num_runs: int
    explain_samples: int
    saliency_samples: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.12, contrast=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, eval_tfms


def create_dataloaders(dataset_root: Path, image_size: int, batch_size: int, num_workers: int = 2):
    train_tfms, eval_tfms = get_transforms(image_size)
    pin_memory = torch.cuda.is_available()

    train_ds = datasets.ImageFolder(dataset_root / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(dataset_root / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(dataset_root / "test", transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def _load_default_weights(backbone: str, pretrained: bool):
    if not pretrained:
        return None

    weight_map = {
        "alexnet": models.AlexNet_Weights.DEFAULT,
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "densenet121": models.DenseNet121_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
    }
    return weight_map[backbone]


def build_model(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
    weights = _load_default_weights(backbone, pretrained)

    if backbone == "alexnet":
        model = models.alexnet(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif backbone == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "densenet121":
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
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
    raise ValueError(f"Unsupported backbone for Grad-CAM: {backbone}")


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / max(total, 1), correct / max(total, 1), f1


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        target = logits[:, class_idx]
        self.model.zero_grad(set_to_none=True)
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations.")

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
        return cam.detach()


def iou_score(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> float:
    a_bin = a >= threshold
    b_bin = b >= threshold
    inter = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    a_std = float(a_flat.std())
    b_std = float(b_flat.std())
    if a_std < 1e-10 or b_std < 1e-10:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def pairwise_stability_metrics(heatmaps: List[np.ndarray]) -> Dict[str, float]:
    if len(heatmaps) < 2:
        return {"iou": float("nan"), "ssim": float("nan"), "pearson": float("nan")}

    ious, ssims, pears = [], [], []
    for i in range(len(heatmaps)):
        for j in range(i + 1, len(heatmaps)):
            a = heatmaps[i]
            b = heatmaps[j]
            ious.append(iou_score(a, b))
            ssims.append(float(ssim(a, b, data_range=1.0)))
            pears.append(pearson_corr(a, b))

    return {
        "iou": float(np.mean(ious)),
        "ssim": float(np.mean(ssims)),
        "pearson": float(np.mean(pears)),
    }


def apply_tta(x: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
    return [
        ("identity", x),
        ("hflip", TF.hflip(x)),
        ("rot_p10", TF.rotate(x, 10.0)),
        ("rot_n10", TF.rotate(x, -10.0)),
        ("bright_1p1", TF.adjust_brightness(x, 1.1)),
    ]


def invert_tta_heatmap(name: str, heatmap: torch.Tensor) -> torch.Tensor:
    if name == "hflip":
        return torch.flip(heatmap, dims=[1])
    if name == "rot_p10":
        return TF.rotate(heatmap.unsqueeze(0), -10.0).squeeze(0)
    if name == "rot_n10":
        return TF.rotate(heatmap.unsqueeze(0), 10.0).squeeze(0)
    return heatmap


def explanation_with_tta(grad_cam: GradCAM, x: torch.Tensor, class_idx: Optional[int]) -> np.ndarray:
    cams = []
    for name, x_aug in apply_tta(x):
        cam = grad_cam.generate(x_aug, class_idx=class_idx)
        cam = invert_tta_heatmap(name, cam)
        cams.append(cam.cpu())

    avg = torch.stack(cams, dim=0).mean(dim=0)
    avg = torch.clamp(avg, 0.0, 1.0)
    return avg.numpy()


def denormalize_image_tensor(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    img = x * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return img


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    h = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 * h - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * h - 1.0) * 1.5, 0.0, 1.0)
    b = np.clip(1.0 - 1.5 * h, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def save_saliency_artifacts(
    out_dir: Path,
    sample_idx: int,
    class_name: str,
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_img = denormalize_image_tensor(image_tensor)
    heat_rgb = heatmap_to_rgb(heatmap)
    overlay = np.clip(0.6 * rgb_img + 0.4 * heat_rgb, 0.0, 1.0)

    stem = f"sample_{sample_idx:04d}_{class_name.replace(' ', '_')}"
    Image.fromarray((rgb_img * 255).astype(np.uint8)).save(out_dir / f"{stem}_input.png")
    Image.fromarray((heat_rgb * 255).astype(np.uint8)).save(out_dir / f"{stem}_heatmap.png")
    Image.fromarray((overlay * 255).astype(np.uint8)).save(out_dir / f"{stem}_overlay.png")


def save_deployment_bundle(
    model: nn.Module,
    bundle_path: Path,
    config: ExperimentConfig,
    class_names: List[str],
) -> None:
    bundle = {
        "model_state_dict": model.state_dict(),
        "backbone": config.backbone,
        "pretrained": config.pretrained,
        "class_names": class_names,
        "image_size": config.image_size,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    torch.save(bundle, bundle_path)


def train_single_run(
    config: ExperimentConfig,
    dataset_root: Path,
    output_dir: Path,
    class_count: int,
    run_seed: int,
    run_id: int,
    use_amp: bool,
):
    set_seed(run_seed)
    device = get_device()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    _, _, _, train_loader, val_loader, test_loader = create_dataloaders(
        dataset_root=dataset_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
    )

    model = build_model(config.backbone, class_count, pretrained=config.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_acc = -1.0
    best_path = output_dir / f"{config.backbone}_run{run_id}_best.pt"

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        tr_loss = total_loss / max(total, 1)
        tr_acc = correct / max(total, 1)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        print(
            f"[{config.backbone}][run {run_id}] epoch {epoch + 1}/{config.epochs} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)

    bundle_path = output_dir / f"{config.backbone}_run{run_id}_deployment.pt"
    save_deployment_bundle(model, bundle_path, config, train_loader.dataset.classes)

    return {
        "model_path": str(best_path),
        "bundle_path": str(bundle_path),
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
    }


def collect_explanation_stability(
    config: ExperimentConfig,
    model_paths: List[str],
    dataset_root: Path,
    class_count: int,
    class_names: List[str],
    saliency_dir: Path,
) -> Dict[str, float]:
    device = get_device()
    _, _, test_ds, _, _, _ = create_dataloaders(
        dataset_root=dataset_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
    )

    explain_count = min(config.explain_samples, len(test_ds))
    indices = list(range(explain_count))

    run_heatmaps_by_sample: List[List[np.ndarray]] = [[] for _ in range(explain_count)]

    for model_path in model_paths:
        model = build_model(config.backbone, class_count, pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        target_layer = get_target_layer(model, config.backbone)
        grad_cam = GradCAM(model, target_layer)

        for local_i, ds_idx in enumerate(indices):
            image, label = test_ds[ds_idx]
            x = image.unsqueeze(0).to(device)
            heatmap = explanation_with_tta(grad_cam, x, class_idx=int(label))
            run_heatmaps_by_sample[local_i].append(heatmap)

        grad_cam.remove()

    saliency_count = min(config.saliency_samples, explain_count)
    if saliency_count > 0:
        for local_i in range(saliency_count):
            ds_idx = indices[local_i]
            image, label = test_ds[ds_idx]
            x = image.unsqueeze(0).to(device)
            sample_heatmaps = run_heatmaps_by_sample[local_i]
            if not sample_heatmaps:
                continue
            avg_heatmap = np.mean(np.stack(sample_heatmaps, axis=0), axis=0)
            class_name = class_names[int(label)]
            save_saliency_artifacts(
                out_dir=saliency_dir,
                sample_idx=ds_idx,
                class_name=class_name,
                image_tensor=x,
                heatmap=avg_heatmap,
            )

    if len(model_paths) < 2:
        print("Stability metrics require num_runs >= 2. Returning NaN for stability fields.")
        return {
            "stability_iou": float("nan"),
            "stability_ssim": float("nan"),
            "stability_pearson": float("nan"),
        }

    sample_metrics = []
    for heatmaps in run_heatmaps_by_sample:
        metrics = pairwise_stability_metrics(heatmaps)
        sample_metrics.append(metrics)

    iou_vals = [m["iou"] for m in sample_metrics]
    ssim_vals = [m["ssim"] for m in sample_metrics]
    pear_vals = [m["pearson"] for m in sample_metrics]

    return {
        "stability_iou": float(np.nanmean(iou_vals)),
        "stability_ssim": float(np.nanmean(ssim_vals)),
        "stability_pearson": float(np.nanmean(pear_vals)),
    }


def run_experiment(
    config: ExperimentConfig,
    dataset_root: Path,
    output_dir: Path,
    class_count: int,
    class_names: List[str],
    base_seed: int,
    use_amp: bool,
):
    config_dir = output_dir / config.backbone / (
        f"lr{config.learning_rate}_wd{config.weight_decay}_bs{config.batch_size}_ep{config.epochs}"
    )
    config_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running config: {asdict(config)} ===")
    run_results = []
    run_rows = []
    model_paths = []
    bundle_paths = []

    for run_id in range(config.num_runs):
        run_seed = base_seed + run_id
        result = train_single_run(config, dataset_root, config_dir, class_count, run_seed, run_id, use_amp=use_amp)
        run_results.append(result)
        run_rows.append(
            {
                "backbone": config.backbone,
                "pretrained": config.pretrained,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "run_id": run_id,
                "run_seed": run_seed,
                "val_acc": result["val_acc"],
                "test_acc": result["test_acc"],
                "test_f1": result["test_f1"],
                "model_path": result["model_path"],
                "bundle_path": result["bundle_path"],
            }
        )
        model_paths.append(result["model_path"])
        bundle_paths.append(result["bundle_path"])

    saliency_dir = config_dir / "saliency_maps"
    stability = collect_explanation_stability(
        config,
        model_paths,
        dataset_root,
        class_count,
        class_names,
        saliency_dir,
    )

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(config_dir / "run_results.csv", index=False)

    return {
        "backbone": config.backbone,
        "pretrained": config.pretrained,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "num_runs": config.num_runs,
        "mean_val_acc": float(np.mean([r["val_acc"] for r in run_results])),
        "mean_test_acc": float(np.mean([r["test_acc"] for r in run_results])),
        "mean_test_f1": float(np.mean([r["test_f1"] for r in run_results])),
        "stability_iou": stability["stability_iou"],
        "stability_ssim": stability["stability_ssim"],
        "stability_pearson": stability["stability_pearson"],
        "saliency_dir": str(saliency_dir),
        "bundle_paths": ";".join(bundle_paths),
        "run_results_path": str(config_dir / "run_results.csv"),
        "run_rows": run_rows,
    }


def write_paper_table(results_df: pd.DataFrame, out_md: Path) -> None:
    cols = [
        "backbone",
        "pretrained",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "epochs",
        "mean_test_acc",
        "mean_test_f1",
        "stability_iou",
        "stability_ssim",
        "stability_pearson",
    ]
    df = results_df[cols].sort_values(by=["backbone", "mean_test_acc"], ascending=[True, False]).copy()

    lines = [
        "# Hyperparameter and Result Table",
        "",
        "| Backbone | Pretrained | LR | Weight Decay | Batch | Epochs | Test Acc | Test F1 | IoU | SSIM | Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt_metric(value: float) -> str:
        if pd.isna(value):
            return "N/A"
        return f"{value:.4f}"

    for _, r in df.iterrows():
        lines.append(
            "| "
            f"{r['backbone']} | {int(bool(r['pretrained']))} | {r['learning_rate']:.1e} | {r['weight_decay']:.1e} | "
            f"{int(r['batch_size'])} | {int(r['epochs'])} | {r['mean_test_acc']:.4f} | {r['mean_test_f1']:.4f} | "
            f"{fmt_metric(r['stability_iou'])} | {fmt_metric(r['stability_ssim'])} | {fmt_metric(r['stability_pearson'])} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Stability-Optimized XAI Pipeline for Lung Disease Detection")
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--backbones", nargs="+", default=["alexnet", "resnet18", "densenet121", "efficientnet_b0"])
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision ImageNet pretrained weights")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[3e-4, 1e-4])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[1e-4, 5e-4])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--epoch-list", nargs="+", type=int, default=None, help="Optional list of epochs to sweep")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use mixed precision on CUDA")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--explain-samples", type=int, default=100)
    parser.add_argument("--saliency-samples", type=int, default=20, help="Number of test samples to save saliency maps for")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-configs", type=int, default=0, help="0 means use all generated configs")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    train_dir = dataset_root / "train"
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    class_count = len(class_names)
    print(f"Detected classes ({class_count}): {class_names}")

    epoch_values = args.epoch_list if args.epoch_list else [args.epochs]
    grid = list(itertools.product(args.backbones, args.learning_rates, args.weight_decays, args.batch_sizes, epoch_values))
    if args.max_configs and args.max_configs > 0:
        grid = grid[: args.max_configs]

    all_rows = []
    all_run_rows = []
    for i, (backbone, lr, wd, bs, epochs) in enumerate(grid):
        config = ExperimentConfig(
            backbone=backbone,
            pretrained=bool(args.pretrained),
            learning_rate=float(lr),
            weight_decay=float(wd),
            batch_size=int(bs),
            epochs=int(epochs),
            image_size=int(args.image_size),
            num_runs=int(args.num_runs),
            explain_samples=int(args.explain_samples),
            saliency_samples=int(args.saliency_samples),
        )

        row = run_experiment(
            config=config,
            dataset_root=dataset_root,
            output_dir=output_dir,
            class_count=class_count,
            class_names=class_names,
            base_seed=args.seed + i * 100,
            use_amp=bool(args.amp),
        )
        run_rows = row.pop("run_rows", [])
        all_run_rows.extend(run_rows)
        all_rows.append(row)

        results_df = pd.DataFrame(all_rows)
        results_df.to_csv(output_dir / "experiment_results.csv", index=False)
        write_paper_table(results_df, output_dir / "paper_results_table.md")

        if all_run_rows:
            pd.DataFrame(all_run_rows).to_csv(output_dir / "experiment_results_per_run.csv", index=False)

    final_df = pd.DataFrame(all_rows)

    print("\n=== Final Results ===")
    print(final_df.sort_values(["backbone", "mean_test_acc"], ascending=[True, False]).to_string(index=False))
    print(f"\nSaved CSV: {output_dir / 'experiment_results.csv'}")
    print(f"Saved table: {output_dir / 'paper_results_table.md'}")


if __name__ == "__main__":
    main()
