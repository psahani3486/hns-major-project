from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path
from collections import defaultdict

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np

from .inference import (
    find_default_bundle_path,
    load_bundle,
    predict_probabilities,
    predict_with_explanation,
    rank_probabilities,
    summarize_probabilities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle_override = os.getenv("MODEL_BUNDLE_PATH")
bundle_path = None
if bundle_override:
    # Support: absolute path, repo-relative path, or remote HTTP/HTTPS URL
    if bundle_override.startswith("http://") or bundle_override.startswith("https://"):
        try:
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            tmpf.close()
            urllib.request.urlretrieve(bundle_override, tmpf.name)
            bundle_path = Path(tmpf.name)
        except Exception as exc:
            bundle_path = None
            _load_error = f"Failed to download MODEL_BUNDLE_PATH URL: {exc}"
    else:
        p = Path(bundle_override)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        bundle_path = p
else:
    try:
        bundle_path = find_default_bundle_path(PROJECT_ROOT)
    except FileNotFoundError:
        bundle_path = None

LOADED = None
_load_error: str | None = None
if bundle_path is not None:
    try:
        if not bundle_path.exists():
            raise FileNotFoundError(str(bundle_path))
        LOADED = load_bundle(bundle_path, DEVICE)
    except Exception as exc:  # keep service up even if bundle loading fails
        LOADED = None
        _load_error = str(exc)
else:
    _load_error = "No deployment bundle found; set MODEL_BUNDLE_PATH (local path or http(s) URL) or add outputs_paper_seed3_ep10/*.pt"


def discover_representative_bundles(project_root: Path) -> dict[str, Path]:
    candidates = sorted(project_root.glob("outputs_paper_seed3_ep10/**/*_deployment.pt"))
    by_backbone: dict[str, list[Path]] = defaultdict(list)
    for candidate in candidates:
        for backbone in ("alexnet", "resnet18", "densenet121", "efficientnet_b0"):
            if backbone in str(candidate).lower():
                by_backbone[backbone].append(candidate)
                break

    selected: dict[str, Path] = {}
    for backbone, paths in by_backbone.items():
        # Prefer run0 if available; otherwise pick first sorted file.
        run0 = [p for p in paths if "run0" in p.stem.lower()]
        selected[backbone] = sorted(run0 or paths)[0]
    return selected


REPRESENTATIVE_BUNDLES = discover_representative_bundles(PROJECT_ROOT)
LOADED_MODELS: dict[str, object] = {}
for name, path in REPRESENTATIVE_BUNDLES.items():
    try:
        LOADED_MODELS[name] = load_bundle(path, DEVICE)
    except Exception:
        # skip models that fail to load; keep server available
        continue

app = FastAPI(title="Lung Disease Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "bundle": str(bundle_path) if bundle_path is not None else None,
        "model_loaded": bool(LOADED),
        "load_error": _load_error,
    }


@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": "Lung Disease Inference API",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "model": "/model",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/model")
def model_info() -> dict[str, object]:
    if LOADED is None:
        return {
            "model_loaded": False,
            "error": _load_error,
            "available_models": [
                {
                    "model_name": model_name,
                    "bundle": str(REPRESENTATIVE_BUNDLES[model_name]),
                }
                for model_name in sorted(REPRESENTATIVE_BUNDLES.keys())
            ],
        }

    return {
        "model_loaded": True,
        "backbone": LOADED.backbone,
        "classes": LOADED.class_names,
        "image_size": LOADED.image_size,
        "bundle": str(bundle_path) if bundle_path is not None else None,
        "available_models": [
            {
                "model_name": model_name,
                "bundle": str(REPRESENTATIVE_BUNDLES[model_name]),
            }
            for model_name in sorted(LOADED_MODELS.keys())
        ],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if LOADED is None:
        raise HTTPException(status_code=503, detail=f"No model loaded: {_load_error}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        result = predict_with_explanation(image_bytes, LOADED, DEVICE)

        per_model_results: list[dict[str, object]] = []
        prob_vectors: list[np.ndarray] = []
        for model_name, loaded_model in sorted(LOADED_MODELS.items()):
            probs = predict_probabilities(image_bytes, loaded_model, DEVICE)
            prob_vectors.append(probs)
            analysis = summarize_probabilities(probs, loaded_model.class_names)
            per_model_results.append(
                {
                    "model_name": model_name,
                    "bundle": str(REPRESENTATIVE_BUNDLES[model_name]),
                    "predicted_index": analysis["predicted_index"],
                    "predicted_label": analysis["predicted_label"],
                    "top_confidence": analysis["top_confidence"],
                    "analysis": analysis,
                    "class_probabilities": rank_probabilities(probs, loaded_model.class_names),
                }
            )

        ensemble_probs = np.stack(prob_vectors, axis=0).mean(axis=0) if prob_vectors else np.array([])
        if ensemble_probs.size:
            result["ensemble_summary"] = {
                "method": "mean_probability",
                "num_models": len(prob_vectors),
                "class_probabilities": rank_probabilities(ensemble_probs, LOADED.class_names),
                "analysis": summarize_probabilities(ensemble_probs, LOADED.class_names),
            }
        result["per_model_results"] = per_model_results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return JSONResponse(content=result)
