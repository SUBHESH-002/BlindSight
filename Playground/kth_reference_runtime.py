from pathlib import Path

import cv2
import numpy as np
import torch


DEVICE = "cpu"
LABELS = {
    0: "Boxing",
    1: "Handclapping",
    2: "Walking",
}
ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "kth_temporal_dataset.pt"


class KTHReferenceModel:
    def __init__(self, features: np.ndarray, labels: np.ndarray, filenames: list[str]):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.filenames = filenames
        self.flat_features = self.features.reshape(len(self.features), -1)
        self.class_ids = np.array(sorted(LABELS), dtype=np.int64)
        self.class_centroids = np.stack(
            [self.flat_features[self.labels == class_id].mean(axis=0) for class_id in self.class_ids]
        )


def load_model():
    if not DATASET_PATH.exists():
        raise RuntimeError(f"KTH temporal dataset not found: {DATASET_PATH}")

    data = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    features = data["features"].detach().cpu().numpy()
    labels = data["labels"].detach().cpu().numpy()
    filenames = list(data.get("filenames", []))
    return KTHReferenceModel(features=features, labels=labels, filenames=filenames)


def predict_frames(model: KTHReferenceModel, frames: np.ndarray, query: str, top_k: int = 3):
    sample = frames_to_temporal_sample(frames).reshape(1, -1)

    centroid_distances = np.linalg.norm(model.class_centroids - sample, axis=1)
    centroid_scores = _distance_scores(centroid_distances)

    k = min(7, len(model.flat_features))
    reference_distances = np.linalg.norm(model.flat_features - sample, axis=1)
    nearest_indices = np.argsort(reference_distances)[:k]

    neighbor_votes = np.zeros(len(model.class_ids), dtype=np.float32)
    for idx in nearest_indices:
        class_position = int(np.where(model.class_ids == model.labels[idx])[0][0])
        neighbor_votes[class_position] += 1.0 / (reference_distances[idx] + 1e-6)
    neighbor_scores = neighbor_votes / (neighbor_votes.sum() + 1e-8)

    scores = (0.65 * neighbor_scores) + (0.35 * centroid_scores)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "label": LABELS[int(model.class_ids[index])],
            "score": float(scores[index]),
            "source": "kth_reference_content",
        }
        for index in top_indices
    ]


def frames_to_temporal_sample(frames: np.ndarray):
    if len(frames) == 0:
        raise ValueError("No frames were provided for prediction.")

    indices = np.linspace(0, len(frames) - 1, 5, dtype=int)
    resized = []

    for index in indices:
        frame = frames[index]
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        if frame.max() > 1.0:
            frame = frame / 255.0
        small = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA)
        resized.append(small)

    return np.stack(resized).astype(np.float32)


def _distance_scores(distances: np.ndarray):
    scale = np.std(distances) + 1e-6
    logits = -(distances - distances.min()) / scale
    exp_logits = np.exp(logits - logits.max())
    return exp_logits / exp_logits.sum()
