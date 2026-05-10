# VL-JEPA Video Predictor UI

This app connects a video upload UI to a FastAPI endpoint for your VL-JEPA-style action predictor.

## Run

```powershell
pip install -r requirements.txt
uvicorn backend:app --reload
```

Open:

```text
http://localhost:8000
```

## Connect Your Model

The backend calls `vl_jepa_runtime.py`.

The backend is configured for your `train_and_save.py` script.

By default, it expects:

```text
saved_model/predictor_best.pt
saved_model/checkpoint_latest.pt
```

Or set a custom saved-model folder:

```powershell
$env:VL_JEPA_SAVE_DIR="C:\path\to\your\saved_model"
uvicorn backend:app --reload
```

The folder should sit next to `test_model.py`, because `vl_jepa_runtime.py` imports your model classes from there.

Add this line in `train_and_save.py` after `text_mapping` is created so inference uses the same labels:

```python
with open(os.path.join(save_dir, "text_mapping.json"), "w", encoding="utf-8") as f:
    json.dump({int(k): v for k, v in text_mapping.items()}, f, indent=2)
```

The runtime rebuilds:

```text
HFVJEPAEncoder
HFEmbeddingGemmaYEncoder
TransformerPredictor
```

Then it loads:

```text
predictor_best.pt -> model.predictor
checkpoint_latest.pt -> Y-encoder projection, if present
text_mapping.json -> class labels, if present
```

## API Response

The frontend expects:

```json
{
  "predictions": [
    { "label": "Picking something up", "score": 0.87 }
  ]
}
```

The backend already returns this shape from `vl_jepa_runtime.py::predict_frames()`.
