---
marp: true
theme: default
paginate: true
---

# 🚀 Project VL-JEPA: Pre-Final Review
## Testing, Validation, Optimization & Results

**Date**: April 24, 2026
**Focus**: Spatio-Temporal Representation Learning for CCTV Actions

*This presentation covers our model's performance on real-world datasets, metric validation, baseline comparisons, and optimization analytics.*

---

# 📊 1. Testing Under Different Datasets

We evaluated the model across diverse data domains to ensure robustness and zero-shot generalization capabilities in real-world CCTV environments.

### KTH Temporal Dataset
- **Composition**: Structured human actions (Walking, Boxing, Handclapping).
- **Processing**: 16x16 frame patching over 5 consecutive frames.
- **Goal**: Establishing baseline accuracy on localized, homogeneous backgrounds.

### Something-Something V2 (SSV2) Dataset
- **Composition**: Complex, object-interaction driven human actions.
- **Goal**: Evaluating the Llama-3.2 Predictor and Gemma-based Text Encoders on highly dynamic spatio-temporal features.

---

# 📈 2. Validation Using Performance Metrics

Rigorous testing has yielded strong quantitative results, particularly highlighting the efficacy of our **Bi-Directional InfoNCE Loss** in zero-shot classification.

- **KTH Unseen Accuracy**: **71.67%** (Achieved across 500 epochs with batched inference).

### Multiclass ROC & AUC Performance
The Receiver Operating Characteristic (ROC) curve validates class-level prediction stability:
- **Area Under Curve (AUC)** consistently **> 0.90** for primary classes (Walking, Boxing, Clapping).
- The tau-scaled (0.1) cosine similarity provides sharpened prediction probabilities.

---

<div align="center">
  <img src="output.png" height="500" />
  <p><i>Multiclass ROC and Training Loss performance</i></p>
</div>

---

# ⚖️ 3. Comparison of Baseline vs Proposed Models

How does **VL-JEPA** stand against conventional architectures?

| Feature | Standard Baseline (e.g., ViT Classification) | Proposed Model (Batched VL-JEPA) |
| :--- | :--- | :--- |
| **Architecture Goal** | Direct Spatial Mapping (Supervised) | Joint Embedding Predictive Architecture |
| **Loss Function** | Cross-Entropy Loss | Bi-Directional InfoNCE Contrastive Loss |
| **Zero-Shot Ability** | ❌ None (Requires Fine-Tuning) | ✅ High (Via Text-Vision Cosine Similarity) |
| **Temporal Context** | Weak (Frame aggregation) | Strong (Spatio-Temporal Sequence Pooling) |

---

# ⚙️ 4. Optimization and Error Analysis

Continuous optimization is paramount to maximizing model throughput while mitigating hardware constraints.

### Optimization Strategies
- **Fully Vectorized Forward/Backward Passes**: We optimized our predictor using batched operations, reshaping sequences to `(Batch, Time, Height, Width, Channels)`.
- **Gradient Freezing**: Vision (`vit-large-patch16-224`) and Text (`GemmaYEncoder`) backbones were frozen to focus gradient updates exclusively on the Predictor, saving VRAM.

### Error Analysis
- **Overlap Misclassifications**: Ambiguous actions introduce slight entropy in our cosine similarity margins.
- **Hardware Bottlenecks**: Heavy dimensionality required careful batch sizing (e.g., batch_size=4) to prevent OOM errors.

---

# 🖥️ 5. Real-World CCTV Interface Demo

Our deployed React frontend successfully captures and processes video feeds in real-time, sending data to the VL-JEPA inference engine.

<div align="center">
  <img src="cctv Ui.webp" height="350" />
</div>

> The system maps detected actions to natural language queries ("Describe the action") instantly predicting: *“A person walking.”*

---

# 🔍 6. Feature Processing & Real Pixels

A closer look at how real-world video frames are represented and patched before entering the JEPA Vision Encoder.

<div align="center">
  <img src="real_pixels.webp" height="350" />
</div>

### Next Steps
- Finalize SSV2 hyperparameter tuning.
- Deploy the optimized ONNX models for edge-inference on actual CCTV cameras.
