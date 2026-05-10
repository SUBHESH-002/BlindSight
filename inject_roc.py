import json

with open('model_2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find index of the SSV2 Markdown cell
target_idx = len(nb['cells'])
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        if any("7. SSV2 DATASET INTEGRATION" in line for line in cell['source']):
            target_idx = i
            break

md_cell = {
   "cell_type": "markdown",
   "id": "roc_curve_md",
   "metadata": {},
   "source": [
    "# ==========================================\n",
    "# MULTICLASS ROC CURVE FOR KTH DATASET\n",
    "# =========================================="
   ]
}

code_source = r"""import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def stable_softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def get_prediction_probs(model, feat_seq, num_classes, text_mapping, vocab):
    # 1. Format Vision Feature for the batch dimension: (Batch=1, Time=5, H=16, W=16, C=3)
    X_v = np.array(feat_seq).reshape(1, 5, 16, 16, 3)
    S_v = model.vision_encoder.forward(X_v) 
    
    # 2. Format Query
    query_indices = np.array([[vocab[w] for w in ["describe", "action"]]])
    S_q_seq = model.text_encoder.get_sequence_embeddings(query_indices)
    
    # 3. Predict Target Embedding
    S_y_hat = model.predictor.forward(S_v, S_q_seq) # Output: (1, target_dim)
    
    # 4. Zero-Shot Classification via Cosine Similarity
    similarities = []
    for c in range(num_classes):
        target_indices = np.array([[vocab[w] for w in text_mapping[c]]])
        S_y = model.text_encoder.encode(target_indices) # Output: (1, target_dim)
        
        sim = np.dot(S_y_hat[0], S_y[0]) / (np.linalg.norm(S_y_hat[0]) * np.linalg.norm(S_y[0]) + 1e-8)
        # Scale by tau=0.1 (as in InfoNCE) to sharpen the probability distribution
        similarities.append(sim / 0.1) 
        
    probs = stable_softmax(np.array(similarities))
    return probs[0]

def plot_multiclass_roc(model, df_test, num_classes, label_names, text_mapping, vocab, feature_col='features'):
    y_true = []
    y_scores = []

    for _, row in df_test.iterrows():
        val = row[feature_col]
        if isinstance(val, str):
            features = np.array(ast.literal_eval(val))
        else:
            features = np.array(val)
            
        true_label = int(row['target_class'])
        probs = get_prediction_probs(model, features, num_classes, text_mapping, vocab)

        y_true.append(true_label)
        y_scores.append(probs)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Binarize labels for multilabel ROC
    y_test_bin = label_binarize(y_true, classes=range(num_classes))
    
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'ROC curve of class {label_names[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - KTH Dataset')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    df_eval = pd.read_csv(r"C:\Users\subhe\OneDrive\Desktop\VL-JEPA\codes\VL-JEPA-base-CCTV\kth_temporal_features.csv")
    feat_col = 'features' if 'features' in df_eval.columns else 'hsv_features'
    
    # REPLICATE THE MANUAL TRAIN/TEST SPLIT
    df_shuffled_full = df_eval.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_shuffled_full) * 0.8)
    test_df = df_shuffled_full.iloc[split_idx:].reset_index(drop=True)

    labels = ["Walking", "Boxing", "Handclapping"]

    eval_text_mapping = {
        0: ["a", "person", "walking"],
        1: ["a", "person", "boxing"],
        2: ["a", "person", "clapping"]
    }

    all_words = ["describe", "action", "a", "person", "walking", "boxing", "clapping"]
    eval_vocab = {word: idx for idx, word in enumerate(set(all_words))}

    print(f"Plotting ROC Curve for {len(test_df)} unseen KTH temporal test samples...")
    plot_multiclass_roc(trained_model, test_df, 3, labels, eval_text_mapping, eval_vocab, feature_col=feat_col)
"""

code_cell = {
   "cell_type": "code",
   "execution_count": None,
   "id": "roc_curve_code",
   "metadata": {},
   "outputs": [],
   "source": [line + "\\n" for line in code_source.split('\\n')]
}

if len(code_cell['source']) > 0:
    code_cell['source'][-1] = code_cell['source'][-1].rstrip('\\n')

nb['cells'].insert(target_idx, md_cell)
nb['cells'].insert(target_idx + 1, code_cell)

with open('model_2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
