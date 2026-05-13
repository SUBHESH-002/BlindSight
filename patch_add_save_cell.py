"""
patch_add_save_cell.py
======================
Adds a 'Save Model for Backend' cell to model-03.ipynb.
This cell exports ALL weights (predictor + text encoder + vision encoder)
so the backend loads them properly and stops using random vision weights.
"""
import json

NOTEBOOK = r"O:\VL-JEPA\codes\VL-JEPA-base-CCTV\model-03.ipynb"

SAVE_CELL_SOURCE = r"""# ==========================================
# SAVE TRAINED MODEL FOR BACKEND
# ==========================================
# Run this cell after training to export ALL weights to Playground/numpy_model.npz.
# The backend will then load BOTH the trained text encoder AND vision encoder.
import numpy as np
import os

def save_model_for_backend(model, vocab, text_mapping,
                           save_path=r"O:\VL-JEPA\codes\VL-JEPA-base-CCTV\Playground\numpy_model.npz"):
    state = {}

    # ── Predictor weights ──────────────────────────────────────────────────
    state["W_proj"] = model.predictor.W_proj
    for i, blk in enumerate(model.predictor.blocks):
        state[f"pred_b{i}_n1g"]   = blk.norm1.gamma
        state[f"pred_b{i}_Wq"]    = blk.attn.W_q
        state[f"pred_b{i}_Wk"]    = blk.attn.W_k
        state[f"pred_b{i}_Wv"]    = blk.attn.W_v
        state[f"pred_b{i}_Wo"]    = blk.attn.W_o
        state[f"pred_b{i}_n2g"]   = blk.norm2.gamma
        state[f"pred_b{i}_Wgate"] = blk.ffn.W_gate
        state[f"pred_b{i}_Wup"]   = blk.ffn.W_up
        state[f"pred_b{i}_Wdown"] = blk.ffn.W_down

    # ── Text encoder weights ───────────────────────────────────────────────
    state["txt_tok"]  = model.text_encoder.token_embeddings
    state["txt_pos"]  = model.text_encoder.position_embeddings
    state["txt_WQ"]   = model.text_encoder.W_Q
    state["txt_WK"]   = model.text_encoder.W_K
    state["txt_WV"]   = model.text_encoder.W_V
    state["txt_WOut"] = model.text_encoder.W_Output

    # ── Vision encoder weights (NEW — needed for deterministic inference) ──
    ve = model.vision_encoder
    state["vis_W_patch"]   = ve.W_patch
    state["vis_pos_embed"] = ve.pos_embed
    for l in range(ve.depth):
        state[f"vis_Wq_{l}"] = ve.Wq[l]
        state[f"vis_Wk_{l}"] = ve.Wk[l]
        state[f"vis_Wv_{l}"] = ve.Wv[l]
        state[f"vis_Wo_{l}"] = ve.Wo[l]
        state[f"vis_W1_{l}"] = ve.W1[l]
        state[f"vis_W2_{l}"] = ve.W2[l]
        state[f"vis_g1_{l}"] = ve.gamma1[l]
        state[f"vis_b1_{l}"] = ve.beta1[l]
        state[f"vis_g2_{l}"] = ve.gamma2[l]
        state[f"vis_b2_{l}"] = ve.beta2[l]

    # ── Metadata ────────────────────────────────────────────────────────────
    state["vocab"]        = np.array(vocab,        dtype=object)
    state["text_mapping"] = np.array(text_mapping, dtype=object)

    np.savez(save_path, **state)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Saved {len(state)} weight arrays -> {save_path}")
    print(f"File size: {size_mb:.1f} MB")

# Call it (make sure 'trained_model', 'vocab', 'text_mapping' exist from training cell)
if "trained_model" in dir():
    save_model_for_backend(trained_model, vocab, text_mapping)
else:
    print("ERROR: trained_model not found. Run the training cell first.")
"""

def add_save_cell():
    with open(NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Check if already exists
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "save_model_for_backend" in src:
                print("save_model_for_backend already exists — skipping")
                return

    # Build new cell
    new_cell = {
        "cell_type": "code",
        "id": "save_model_backend_cell",
        "metadata": {},
        "source": SAVE_CELL_SOURCE,
        "outputs": [],
        "execution_count": None,
    }

    nb["cells"].append(new_cell)

    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Added 'Save Trained Model for Backend' cell to {NOTEBOOK}")
    print("Run this cell in Jupyter AFTER training to export all weights including the vision encoder.")

if __name__ == "__main__":
    add_save_cell()
