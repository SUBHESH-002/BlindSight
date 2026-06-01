# About the VL-JEPA CCTV Inference Project

## Project Overview
The VL-JEPA (Video-Language Joint Embedding Predictive Architecture) project is a highly specialized machine learning system designed to perform spatio-temporal representation learning for real-time CCTV action recognition. The system supports multi-class video action recognition (e.g., Walking, Boxing, Handclapping). 

A key capability of this system is its zero-shot generalization through the use of a Bi-Directional InfoNCE Contrastive Loss. It matches the spatio-temporal features of short video clips with text embeddings of descriptive prompts to predict actions, without traditional supervised classification heads.

The project is divided into three primary domains:
1. **Model Training & Research Data**: Includes Jupyter Notebooks and dataset preparation scripts.
2. **FastAPI Backend (`Playground/`)**: Hosts the inference engine, running the custom NumPy-based transformer implementation to predict actions on live video streams or uploaded clips.
3. **React Frontend (`frontend/`)**: Provides a cyberpunk/high-tech UI for simulating real-world CCTV tracking, capturing webcam frames or video uploads and sending them to the backend for real-time inference.

## Architecture & Components

### 1. Data Preparation & Feature Extraction
- **`rebuild_dataset_from_sequences.py`**: A robust script that downloads the official KTH dataset `00sequences.txt` label files, parses the frame boundaries for different action sequences, and builds the capped `kth_temporal_features.csv` dataset.
- **`scratch.py`**: Contains utility functions to process raw video files, apply mock CCTV occlusion masks, and prepare the 5-frame sequence patches for the Vision Encoder.

### 2. Core Model Implementation
- **`extracted_final.py`**: The exported core implementation from the final Jupyter Notebook. Contains the custom `NumPyVJEPAEncoder` (Vision Encoder), `BatchedSelfAttentionYEncoder` (Text Encoder), and `BatchedTransformerJEPA` predictor architecture.
- **`resumable_training_loop.py`**: Contains the training loop supporting checkpoints and evaluation functions for the model.

### 3. Backend Inference Server (`Playground/`)
- **`backend.py`**: The FastAPI server that exposes a WebSocket endpoint (`/ws/predict`) for live webcam feeds and a standard HTTP POST endpoint (`/predict/video`) for video file uploads.
- **`numpy_jepa_runtime.py` / `model_adapter.py`**: Pure NumPy runtime implementations of the model used in production to avoid heavy PyTorch dependencies during inference. They load weights from `numpy_model.npz`.

### 4. Frontend Interface (`frontend/`)
- A modern React application powered by Vite and styled with Tailwind CSS. It features a responsive layout, a timeline for tracking events, and a futuristic "HUD" display representing real-time object tracking confidence levels.

## Unnecessary / Deprecated Files

The root directory contains several files that represent experimental checkpoints, outdated scripts, or one-off development utilities that are no longer essential to the final application:

- **`temp.ipynb`, `model.ipynb`, `model_2.ipynb`, `model-03.ipynb`**: Outdated iterations of the model notebooks. The final logic resides in `Final_model.ipynb` and `extracted_final.py`.
- **`notebook_dump.py` & `notebook_dump_2.py`**: Temporary Python dumps generated from Jupyter notebooks.
- **`extract.py`**: A very simple script used once to extract code from `Final_model.ipynb`.
- **`patch_add_save_cell.py`**: A one-off utility script used to patch a specific Jupyter notebook by dynamically adding a cell to export the model state.
- **`inject_roc.py`**: A one-off script used to inject ROC curve plotting code into an intermediate notebook (`model_2.ipynb`).
- **`frontend.txt`**: A development prompt text file used to instruct an AI on how to build the React frontend. It serves as historical context but is not executed by any system.
