import importlib
import os
from pathlib import Path

import numpy as np


class ModelNotReadyError(RuntimeError):
    pass


class VLJEPAPredictor:
    def __init__(self):
        self.runtime = None
        self.model = None
        self.device = "unknown"
        self.status_message = "Model runtime has not been loaded yet."

    @property
    def is_ready(self):
        return self.runtime is not None

    def load(self):
        if self.is_ready:
            return

        module_name = os.environ.get("VL_JEPA_RUNTIME_MODULE", "kth_reference_runtime")

        try:
            runtime = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ModelNotReadyError(
                "Model files are not installed on this device yet. Expected a local "
                "runtime like vl_jepa_runtime.py plus model files such as test_model.py "
                "and saved_model/. You can also set VL_JEPA_RUNTIME_MODULE to another "
                "local runtime module."
            ) from exc
        except Exception as exc:
            raise ModelNotReadyError(
                f"Found runtime module '{module_name}', but it could not be loaded: {exc}"
            ) from exc

        if not hasattr(runtime, "load_model"):
            raise ModelNotReadyError(f"{module_name} must define load_model().")

        if not hasattr(runtime, "predict_frames"):
            raise ModelNotReadyError(
                f"{module_name} must define predict_frames(model, frames, query, top_k)."
            )

        try:
            model = runtime.load_model()
        except Exception as exc:
            raise ModelNotReadyError(str(exc)) from exc

        self.runtime = runtime
        self.model = model
        self.device = getattr(runtime, "DEVICE", "cuda/cpu")
        if model is None:
            self.status_message = "Running in mock mode."
        else:
            self.status_message = f"Loaded runtime module: {module_name}"

    def predict_video(
        self,
        video_path: Path,
        query: str,
        top_k: int = 1,
        filename: str | None = None,
    ):
        self.load()

        segments = extract_segments(video_path)
        all_predictions = []

        for frames, timestamp in segments:
            preds = self.runtime.predict_frames(
                self.model,
                frames=frames,
                query=query,
                top_k=top_k,
            )

            preds = normalize_predictions(preds, top_k=top_k)

            if preds:
                best = preds[0]
                all_predictions.append(
                    {
                        "time": format_time(timestamp),
                        "label": best["label"],
                        "score": best["score"],
                        "source": best.get("source", self.status_message),
                    }
                )

        preview_frames, duration = extract_preview_frames(video_path)

        return {
            "predictions": all_predictions,
            "preview_frames": preview_frames,
            "duration": duration,
        }


def extract_segments(video_path: Path, segment_size=8):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    segments = []
    index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256))
        frames.append(frame)

        if len(frames) == segment_size:
            timestamp = index / fps
            segments.append((np.stack(frames), timestamp))
            frames = []

        index += 1

    cap.release()
    return segments


def format_time(seconds: float):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def extract_preview_frames(video_path: Path, max_frames=140):
    import base64
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else 0
    step = max(1, int(total_frames / max_frames)) if total_frames else max(1, int(fps / 4))
    frames = []
    index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if index % step == 0:
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if success:
                frames.append(
                    {
                        "time": round(index / fps, 3),
                        "image": base64.b64encode(encoded).decode("ascii"),
                    }
                )

        index += 1

    cap.release()
    return frames, duration


def normalize_predictions(predictions, top_k: int):
    normalized = []

    for item in list(predictions)[:top_k]:
        if isinstance(item, dict):
            label = item.get("label") or item.get("action") or "Unknown action"
            score = item.get("score", item.get("confidence", 0.0))
        else:
            label, score = item

        score = float(score)
        if score > 1:
            score = score / 100

        normalized.append({"label": str(label), "score": max(0.0, min(score, 1.0))})

    return normalized


_PREDICTOR = VLJEPAPredictor()


def get_predictor():
    if not _PREDICTOR.is_ready:
        try:
            _PREDICTOR.load()
        except Exception as exc:
            _PREDICTOR.status_message = str(exc)
    return _PREDICTOR
