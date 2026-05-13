"""
Patch script: replaces predict_frames (and everything after it) in
numpy_jepa_runtime.py with a hybrid optical-flow + model-score classifier.
Run from inside Playground/:
    python _patch_runtime.py
"""

NEW_FUNCTIONS = r'''def predict_frames(model, frames, query: str = "describe action", top_k: int = 5):
    """
    Called periodically by the frontend.
    frames shape from frontend: (8, 256, 256, 3) RGB uint8

    The vision encoder weights are NOT saved in numpy_model.npz (only the
    predictor and text encoder are persisted), so running frames through the
    vision encoder produces biased random outputs.
    We blend model text-alignment scores (20%) with optical-flow heuristics (80%).
    """
    # -- 1. Resize frames and run through vision encoder + predictor ----------
    resized_frames = []
    indices = np.linspace(0, len(frames) - 1, 5, dtype=int)
    for idx in indices:
        small = cv2.resize(frames[idx], (16, 16), interpolation=cv2.INTER_AREA)
        resized_frames.append(small)

    X_v = np.array(resized_frames) / 255.0
    X_v = X_v.reshape(1, 5, 16, 16, 3)

    S_v = model.vision_encoder.forward(X_v)

    words = ["describe", "action"]
    query_indices = np.array([[model.vocab.get(w, 0) for w in words]])
    S_q_seq = model.text_encoder.get_sequence_embeddings(query_indices)

    S_y_hat = model.predictor.forward(S_v, S_q_seq)

    similarities = []
    for c in range(model.num_classes):
        target_tokens = [model.vocab.get(w, 0) for w in model.text_mapping[c]]
        S_y = model.text_encoder.encode(np.array([target_tokens]))
        sim = float(
            np.dot(S_y_hat[0], S_y[0])
            / (np.linalg.norm(S_y_hat[0]) * np.linalg.norm(S_y[0]) + 1e-8)
        )
        similarities.append(sim)

    similarities = np.array(similarities)
    exp_sim = np.exp(similarities * 10)
    model_scores = exp_sim / np.sum(exp_sim)

    # -- 2. Optical-flow heuristic -------------------------------------------
    motion_scores = _motion_heuristic_scores(frames, model)

    # -- 3. Blend: 20% text-alignment, 80% real motion -----------------------
    ALPHA = 0.20
    final_scores = ALPHA * model_scores + (1.0 - ALPHA) * motion_scores
    scores = final_scores.tolist()

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "label": " ".join(model.text_mapping[idx]).title(),
            "score": float(scores[idx]),
        }
        for idx in top_indices
    ]


def _motion_heuristic_scores(frames, model):
    """
    Compute class probabilities from dense optical-flow features.
    Returns array of length model.num_classes (order: boxing=0, walking=1, clapping=2).

    KTH motion signatures
    ---------------------
    Walking     - steady moderate magnitude, VERTICAL flow dominant,
                  body translates so LOW bilateral L-R symmetry,
                  LOW temporal variance (smooth motion)
    Boxing      - HIGH magnitude, HORIZONTAL bursts, HIGH temporal variance
    Handclapping - LOW overall magnitude (<2 px/frame), HIGH bilateral symmetry
    """
    if len(frames) < 2:
        return np.ones(model.num_classes) / model.num_classes

    # Dense optical flow between consecutive frames
    flow_vecs = []
    prev_gray = None
    for frame in frames:
        gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_vecs.append(flow)
        prev_gray = gray

    flow_stack = np.stack(flow_vecs, axis=0)   # (T-1, H, W, 2)
    fx = flow_stack[..., 0]                     # horizontal
    fy = flow_stack[..., 1]                     # vertical
    mag = np.sqrt(fx**2 + fy**2)

    mean_mag    = float(np.mean(mag))
    mean_abs_fx = float(np.mean(np.abs(fx)))
    mean_abs_fy = float(np.mean(np.abs(fy)))

    frame_mags = mag.mean(axis=(1, 2))          # per-frame scalar
    temp_var   = float(np.var(frame_mags))

    # Bilateral symmetry: left-half flow vs mirrored right-half
    W = mag.shape[2]
    mid   = W // 2
    left  = mag[:, :, :mid]
    right = mag[:, :, W - mid:][:, :, ::-1]
    symmetry = 1.0 - float(
        np.mean(np.abs(left - right)) / (np.mean(left + right) + 1e-6)
    )
    symmetry = float(np.clip(symmetry, 0.0, 1.0))

    # --- Walking score -------------------------------------------------------
    # Needs: vertical > horizontal, moderate magnitude, smooth over time
    vert_ratio   = mean_abs_fy / (mean_abs_fx + mean_abs_fy + 1e-6)
    movement     = float(np.clip(mean_mag / 3.0, 0.0, 1.0))   # saturates at 3 px/f
    steady_bonus = float(np.clip(1.0 - temp_var / 0.5, 0.0, 1.0))
    walking_score = vert_ratio * movement * steady_bonus

    # --- Boxing score --------------------------------------------------------
    # Needs: horizontal > vertical, high magnitude, high temporal variance
    horiz_ratio  = mean_abs_fx / (mean_abs_fx + mean_abs_fy + 1e-6)
    power        = float(np.clip(mean_mag / 5.0, 0.0, 1.0))   # saturates at 5 px/f
    burst_bonus  = float(np.clip(temp_var / 0.5, 0.0, 1.0))
    boxing_score = horiz_ratio * power * burst_bonus

    # --- Clapping score ------------------------------------------------------
    # Needs: very low overall magnitude AND high bilateral symmetry
    low_mag_gate   = float(np.clip(1.0 - mean_mag / 2.0, 0.0, 1.0))
    clapping_score = symmetry * low_mag_gate

    raw = np.array([boxing_score, walking_score, clapping_score], dtype=np.float64)

    if raw.sum() < 1e-6:
        return np.ones(model.num_classes) / model.num_classes

    # Temperature-scaled softmax (1.5 = decisive but not argmax)
    temperature = 1.5
    raw_t = raw / temperature
    exp_r = np.exp(raw_t - raw_t.max())
    return exp_r / exp_r.sum()
'''

import pathlib

target = pathlib.Path("numpy_jepa_runtime.py")
content = target.read_text(encoding="utf-8")

marker = "def predict_frames("
idx = content.find(marker)
if idx == -1:
    print("ERROR: predict_frames not found in file!")
else:
    new_content = content[:idx] + NEW_FUNCTIONS + "\n"
    target.write_text(new_content, encoding="utf-8")
    line_count = new_content.count("\n")
    print(f"Done. File rewritten. Lines: {line_count}")
    print(f"predict_frames starts at char {idx}")
