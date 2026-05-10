"""
Backend runtime for the VL-JEPA model trained by train_and_save.py.

Expected project files:
  - test_model.py
  - saved_model/predictor_best.pt or saved_model/checkpoint_latest.pt

Recommended extra file:
  - saved_model/text_mapping.json
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = Path(os.environ.get("VL_JEPA_SAVE_DIR", "saved_model"))
TARGET_DIM = int(os.environ.get("VL_JEPA_TARGET_DIM", "256"))
VISION_MODEL = os.environ.get("VL_JEPA_VISION_MODEL", "facebook/vjepa2-vitl-fpc64-256")
TEXT_MODEL = os.environ.get("VL_JEPA_TEXT_MODEL", "google/embeddinggemma-300m")

_CLASS_EMBEDDINGS = None
_TEXT_MAPPING = None


def get_model_components():
    try:
        from test_model import (
            ACTION_TEXT_MAPPING,
            HFEmbeddingGemmaYEncoder,
            HFVJEPAEncoder,
            TransformerPredictor,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing test_model.py on this device. Place test_model.py next to "
            "vl_jepa_runtime.py, or set VL_JEPA_RUNTIME_MODULE to another local runtime."
        ) from exc

    return ACTION_TEXT_MAPPING, HFEmbeddingGemmaYEncoder, HFVJEPAEncoder, TransformerPredictor


def load_model():
    """
    Rebuild the model components, then load weights saved by train_and_save.py.
    """
    global _TEXT_MAPPING

    if not SAVE_DIR.exists():
        raise RuntimeError(
            f"Save directory not found: {SAVE_DIR}. "
            "Set VL_JEPA_SAVE_DIR to the folder containing predictor_best.pt."
        )

    (
        _action_text_mapping,
        HFEmbeddingGemmaYEncoder,
        HFVJEPAEncoder,
        TransformerPredictor,
    ) = get_model_components()

    text_mapping = load_text_mapping(SAVE_DIR)
    _TEXT_MAPPING = text_mapping

    print("Loading VL-JEPA backend components...")
    vision_encoder = HFVJEPAEncoder(VISION_MODEL)
    text_encoder = HFEmbeddingGemmaYEncoder(target_dim=TARGET_DIM, model_name=TEXT_MODEL)

    predictor = TransformerPredictor(
        target_dim=TARGET_DIM,
        input_embed_dim=vision_encoder.hidden_dim,
        text_embed_dim=text_encoder.hidden_size,
        num_layers=1,
        num_heads=8,
    )

    predictor.to(DEVICE)
    load_saved_weights(SAVE_DIR, predictor, text_encoder)

    for param in vision_encoder.model.parameters():
        param.requires_grad = False
    for param in text_encoder.model.parameters():
        param.requires_grad = False
    for param in predictor.parameters():
        param.requires_grad = False

    vision_encoder.model.eval()
    text_encoder.model.eval()
    predictor.eval()

    return SimpleNamespace(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        predictor=predictor,
        text_mapping=text_mapping,
        num_classes=len(text_mapping),
    )


def load_saved_weights(save_dir: Path, predictor, text_encoder):
    best_path = save_dir / "predictor_best.pt"
    latest_path = save_dir / "checkpoint_latest.pt"

    if best_path.exists():
        print(f"Loading predictor weights: {best_path}")
        predictor_state = torch.load(best_path, map_location=DEVICE, weights_only=False)
        predictor.load_state_dict(predictor_state)
    elif latest_path.exists():
        print(f"Loading latest checkpoint predictor weights: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=DEVICE, weights_only=False)
        predictor.load_state_dict(checkpoint["predictor_state"])
    else:
        raise RuntimeError(
            f"No predictor weights found in {save_dir}. Expected predictor_best.pt "
            "or checkpoint_latest.pt."
        )

    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=DEVICE, weights_only=False)
        projection_state = checkpoint.get("y_encoder_projection_state")

        if projection_state is not None and getattr(text_encoder, "W_Output", None) is not None:
            print("Loading Y-encoder projection weights from checkpoint_latest.pt")
            text_encoder.W_Output.load_state_dict(projection_state)


def load_text_mapping(save_dir: Path):
    action_text_mapping, *_ = get_model_components()
    mapping_path = save_dir / "text_mapping.json"

    if mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        return {int(key): value for key, value in raw.items()}

    print(
        "WARNING: saved_model/text_mapping.json not found. "
        "Falling back to ACTION_TEXT_MAPPING from test_model.py."
    )
    return {int(key): value for key, value in action_text_mapping.items()}


def predict_frames(model, frames: np.ndarray, query: str, top_k: int = 5):
    """
    frames: [T, H, W, C], RGB uint8, already resized by model_adapter.py.
    """
    global _CLASS_EMBEDDINGS

    with torch.no_grad():
        video = torch.from_numpy(frames).float()

        if video.max() > 1:
            video = video / 255.0

        video = video.unsqueeze(0).permute(0, 1, 4, 2, 3).to(model.vision_encoder.device)

        s_v = model.vision_encoder.forward(video)
        s_q_seq = model.text_encoder.get_sequence_embeddings([query or "Describe the action."])
        s_y_hat = model.predictor(s_v, s_q_seq)

        s_y_hat = torch.nan_to_num(s_y_hat, nan=0.0, posinf=1.0, neginf=-1.0)

        if _CLASS_EMBEDDINGS is None:
            class_texts = [model.text_mapping[index] for index in range(model.num_classes)]
            _CLASS_EMBEDDINGS = model.text_encoder.encode(class_texts)

        class_embeddings = _CLASS_EMBEDDINGS.to(
            s_y_hat.device,
            dtype=s_y_hat.dtype,
        )
        class_embeddings = torch.nan_to_num(
            class_embeddings,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        similarities = torch.matmul(
            F.normalize(s_y_hat, p=2, dim=1, eps=1e-6),
            F.normalize(class_embeddings, p=2, dim=1, eps=1e-6).t(),
        ).squeeze(0)

        scores = torch.softmax(similarities, dim=0)
        values, indices = torch.topk(scores, k=min(top_k, model.num_classes))

    return [
        {
            "label": model.text_mapping[int(index.item())],
            "score": float(value.item()),
        }
        for value, index in zip(values, indices)
    ]
