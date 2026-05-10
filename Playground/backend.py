import shutil
import tempfile
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from model_adapter import ModelNotReadyError, get_predictor, normalize_predictions


ROOT = Path(__file__).resolve().parent

app = FastAPI(title="VL-JEPA Video Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/styles.css")
def styles():
    return FileResponse(ROOT / "styles.css", media_type="text/css")


@app.get("/app.js")
def script():
    return FileResponse(ROOT / "app.js", media_type="application/javascript")


app.mount("/static", StaticFiles(directory=ROOT), name="static")


@app.get("/health")
def health():
    predictor = get_predictor()
    return {
        "ok": True,
        "model_ready": predictor.is_ready,
        "device": predictor.device,
        "message": predictor.status_message,
    }


@app.post("/predict")
async def predict_legacy(
    video: Annotated[UploadFile, File()],
    query: Annotated[str, Form()] = "Predict action",
    top_k: Annotated[int, Form()] = 5,
):
    """Legacy Predict Endpoint"""
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        shutil.copyfileobj(video.file, temp_file)

    try:
        predictor = get_predictor()
        result = predictor.predict_video(
            video_path=temp_path,
            query=query,
            top_k=top_k,
            filename=video.filename,
        )

        if isinstance(result, dict) and "predictions" in result:
            return result

        return {"predictions": result}
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/predict/video")
async def predict_video_events(
    video: Annotated[UploadFile, File()],
    query: Annotated[str, Form()] = "Predict action",
    top_k: Annotated[int, Form()] = 1,
):
    """New endpoint mapped to the React App for Upload Mode"""
    try:
        result = await predict_legacy(video, query, top_k)
        # Map predictions to events
        events = result.get("predictions", [])
        return {
            "events": events,
            "preview_frames": result.get("preview_frames", []),
            "duration": result.get("duration", 0),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    buffer = []
    
    predictor = get_predictor()
    
    try:
        while True:
            data = await websocket.receive_json()
            image_b64 = data.get("image", "")
            query = data.get("prompt", "Predict action")
            
            if not image_b64:
                continue
                
            img_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256))
            
            buffer.append(frame)
            if len(buffer) > 8:
                buffer.pop(0)
                
            frames = np.stack(buffer) if len(buffer) == 8 else np.stack([frame] * 8)
            
            try:
                predictor.load()
                preds = predictor.runtime.predict_frames(
                    predictor.model,
                    frames=frames,
                    query=query,
                    top_k=1,
                )
                preds = normalize_predictions(preds, top_k=1)
                
                action = preds[0]["label"] if preds else "Unknown"
                confidence = preds[0]["score"] if preds else 0.0
                
            except Exception as e:
                print(f"Prediction Error: {e}")
                action = "Prediction Failed"
                confidence = 0.0

            await websocket.send_json({
                "action": action,
                "confidence": confidence,
                "timestamp": data.get("timestamp", "LIVE"),
                "occluded_action": action
            })
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
