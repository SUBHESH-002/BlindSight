# VL-JEPA CCTV Inference Project

This project contains the backend API and frontend interface for running multi-class video action recognition (Boxing, Handclapping, Walking) using a VL-JEPA model.

## Prerequisites

Before setting up the project, ensure you have the following installed:
- [Python 3.10+](https://www.python.org/downloads/)
- [Node.js (v18+) and npm](https://nodejs.org/)
- Git

## 📥 Required Model Weights

Due to size limitations, the model weights and datasets are **not** stored in this Git repository. You must manually download them before running the server.

1. Download `kth_temporal_dataset.pt` from **[sha256:651abe91de1b0e13b3813cc56471e558d2125588fa1a82d433e2ba2597662471](https://github.com/SUBHESH-002/BlindSight/releases/download/v1.0.0/kth_temporal_dataset.pt)** and place it in the **root** folder...
2. *(Optional)* Download `numpy_model.npz` from **[sha256:9cb28ed0f6a2cbb2184d0badc27979e9898d17f89ccf57d3c7f5b57ed7588c1a](https://github.com/SUBHESH-002/BlindSight/releases/download/v1.0.0/numpy_model.npz)** and place it inside the `Playground/` folder...


---

## ⚙️ 1. Backend Setup

The backend is built with FastAPI and runs the model inference.

1. **Open a terminal in the root folder** (`VL-JEPA-base-CCTV/`).
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```
3. **Activate the environment:**
   - **Windows:** `.\.venv\Scripts\Activate.ps1` (or `Activate.bat` for CMD)
   - **Mac/Linux:** `source .venv/bin/activate`
4. **Install backend dependencies:**
   ```bash
   pip install -r Playground/requirements.txt
   ```
5. **Start the backend server:**
   ```bash
   cd Playground
   uvicorn backend:app --reload --port 8000
   ```
   *The server will be running at `http://localhost:8000`.*

---

## 🎨 2. Frontend Setup

The frontend is a React application powered by Vite.

1. **Open a second terminal window** and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. **Install frontend dependencies:**
   ```bash
   npm install
   ```
3. **Start the development server:**
   ```bash
   npm run dev
   ```
   *The UI will be accessible at `http://localhost:5173`.*

---

## 🚀 Running the App

Once both the backend and frontend servers are running:
1. Open your browser and go to `http://localhost:5173`.
2. Upload a short video clip.
3. The backend will process the video and return the classified actions (e.g., Boxing, Handclapping, Walking) along with a preview.