"""
Inference layer.

If a trained model (model/lung_cnn.h5) is present, real CNN inference is used.
Otherwise the app runs in DEMO MODE: a deterministic, image-statistics-based
heuristic produces varied, plausible-looking class probabilities purely so the
dashboard is fully explorable without a trained model / dataset on hand.

DEMO MODE PREDICTIONS ARE NOT MEDICALLY MEANINGFUL. The API always reports
which mode produced a result via the "mode" field so the frontend can display
an honest disclaimer.
"""

import os
import hashlib
import numpy as np
from PIL import Image

from .cnn_model import CLASS_NAMES, IMG_SIZE, MODEL_PATH

_model = None
_model_load_attempted = False


def _try_load_model():
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    if os.path.exists(MODEL_PATH):
        try:
            from tensorflow.keras.models import load_model
            _model = load_model(MODEL_PATH)
        except Exception as e:
            print(f"[predict] Could not load trained model ({e}); using demo mode.")
            _model = None
    return _model


def is_real_model_loaded():
    return _try_load_model() is not None


def _preprocess(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32")
    return img, arr


def _demo_mode_predict(image_path, arr):
    """Heuristic, non-diagnostic stand-in used only when no trained weights exist.
    Uses simple image statistics (texture variance, mean intensity, edge density)
    combined with a stable hash of the file so results are reproducible per image,
    then maps them into a 3-way softmax-like distribution."""
    gray = arr.mean(axis=2)
    intensity = gray.mean() / 255.0
    texture = gray.std() / 255.0
    edges = np.abs(np.diff(gray, axis=0)).mean() / 255.0

    with open(image_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    seed = int(digest[:8], 16)
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, 0.05, size=3)

    logits = np.array([
        0.9 - texture * 1.4 - edges * 0.6,       # benign: smoother, more uniform
        0.4 + texture * 1.1 + intensity * 0.3,   # adenocarcinoma
        0.3 + edges * 1.3 + texture * 0.5,       # squamous cell carcinoma
    ]) + jitter

    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return probs


def predict(image_path):
    model = _try_load_model()
    img, arr = _preprocess(image_path)

    if model is not None:
        batch = np.expand_dims(arr, axis=0)
        probs = model.predict(batch, verbose=0)[0]
        mode = "trained_model"
    else:
        probs = _demo_mode_predict(image_path, arr)
        mode = "demo_heuristic"

    order = np.argsort(probs)[::-1]
    ranked = [
        {"label": CLASS_NAMES[i], "confidence": round(float(probs[i]) * 100, 2)}
        for i in order
    ]
    top = ranked[0]
    malignant_prob = float(probs[1] + probs[2]) * 100  # adenocarcinoma + squamous

    return {
        "mode": mode,
        "predicted_class": top["label"],
        "confidence": top["confidence"],
        "malignant_probability": round(malignant_prob, 2),
        "class_probabilities": ranked,
        "image_size": IMG_SIZE,
    }
