import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def predict(model, feat_seq, num_classes, text_mapping, vocab):
    X_v = np.array(feat_seq).reshape(1, 5, 16, 16, 3)
    S_v = model.vision_encoder.forward(X_v)

    query_indices = np.array([[vocab[w] for w in ["describe", "action"]]])
    S_q_seq = model.text_encoder.get_sequence_embeddings(query_indices)

    S_y_hat = model.predictor.forward(S_v, S_q_seq)

    distances = []
    for c in range(num_classes):
        target_indices = np.array([[vocab[w] for w in text_mapping[c]]])
        S_y = model.text_encoder.encode(target_indices)
        sim = np.dot(S_y_hat[0], S_y[0]) / (
            np.linalg.norm(S_y_hat[0]) * np.linalg.norm(S_y[0]) + 1e-8
        )
        distances.append(-sim)

    return int(np.argmin(distances))


def collect_numpy_state(obj, prefix="model"):
    state = {}

    if isinstance(obj, np.ndarray):
        state[prefix] = obj.copy()
        return state

    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            state.update(collect_numpy_state(value, f"{prefix}.{index}"))
        return state

    if not hasattr(obj, "__dict__"):
        return state

    for name, value in obj.__dict__.items():
        if name == "cache":
            continue
        key = f"{prefix}.{name}"
        if isinstance(value, np.ndarray):
            state[key] = value.copy()
        elif isinstance(value, (list, tuple)) or hasattr(value, "__dict__"):
            state.update(collect_numpy_state(value, key))

    return state


def restore_numpy_state(obj, state, prefix="model"):
    if isinstance(obj, np.ndarray) or not hasattr(obj, "__dict__"):
        return

    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            restore_numpy_state(value, state, f"{prefix}.{index}")
        return

    for name, value in obj.__dict__.items():
        if name == "cache":
            continue
        key = f"{prefix}.{name}"
        if isinstance(value, np.ndarray) and key in state:
            setattr(obj, name, state[key].copy())
        elif isinstance(value, (list, tuple)) or hasattr(value, "__dict__"):
            restore_numpy_state(value, state, key)


def save_training_checkpoint(
    model,
    text_mapping,
    vocab,
    epoch,
    loss_history,
    checkpoint_path="Playground/kth_training_checkpoint.npz",
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        checkpoint_path,
        epoch=int(epoch),
        model_state=collect_numpy_state(model),
        text_mapping=text_mapping,
        vocab=vocab,
        loss_history=np.array(loss_history, dtype=np.float64),
    )
    print(f"Checkpoint saved: epoch {epoch} -> {checkpoint_path}")


def load_training_checkpoint(model, checkpoint_path="Playground/kth_training_checkpoint.npz"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return 0, []

    checkpoint = np.load(checkpoint_path, allow_pickle=True)
    restore_numpy_state(model, checkpoint["model_state"].item())
    start_epoch = int(checkpoint["epoch"])
    loss_history = checkpoint["loss_history"].tolist()
    print(f"Resuming from checkpoint: epoch {start_epoch} <- {checkpoint_path}")
    return start_epoch, loss_history


def save_numpy_model(model, text_mapping, vocab, save_path="Playground/kth_numpy_model_final.npz"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        model_state=collect_numpy_state(model),
        W_proj=model.predictor.W_proj,
        text_mapping=text_mapping,
        vocab=vocab,
    )
    print(f"Final model saved to {save_path}")


def run_training_and_plot(
    csv_path,
    epochs=500,
    batch_size=4,
    num_classes=3,
    checkpoint_path="Playground/kth_training_checkpoint.npz",
    checkpoint_every=5,
    resume=True,
):
    print(f"Loading features from {csv_path}...")
    df = pd.read_csv(csv_path)
    feature_col = "features" if "features" in df.columns else "hsv_features"
    df[feature_col] = df[feature_col].apply(ast.literal_eval)

    df_shuffled_full = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_shuffled_full) * 0.8)
    train_df = df_shuffled_full.iloc[:split_idx].reset_index(drop=True)
    test_df = df_shuffled_full.iloc[split_idx:].reset_index(drop=True)

    text_mapping = {
        0: ["a", "person", "walking"],
        1: ["a", "person", "boxing"],
        2: ["a", "person", "clapping"],
    }

    all_words = ["describe", "action", "a", "person", "walking", "boxing", "clapping"]
    vocab = {word: idx for idx, word in enumerate(all_words)}
    vocab_size = len(vocab)

    embed_dim = 256
    target_dim = 64

    text_encoder = BatchedSelfAttentionYEncoder(
        vocab_size=vocab_size,
        max_seq_len=10,
        embed_dim=embed_dim,
        target_dim=target_dim,
    )
    vision_encoder = NumPyVJEPAEncoder(
        frame_size=16,
        patch_size=4,
        frames=5,
        embed_dim=embed_dim,
        depth=2,
        heads=4,
    )
    model = BatchedTransformerJEPA(
        embed_dim=embed_dim,
        target_dim=target_dim,
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        num_heads=4,
        num_layers=2,
        lr=0.001,
    )

    start_epoch = 0
    loss_history = []
    if resume:
        start_epoch, loss_history = load_training_checkpoint(model, checkpoint_path)

    if start_epoch >= epochs:
        print(f"Checkpoint is already at epoch {start_epoch}; requested epochs={epochs}.")
        return model, text_mapping, vocab, loss_history

    print(f"Starting VL-JEPA Predictor Training on {len(train_df)} samples...")
    print(f"Training from epoch {start_epoch + 1} to {epochs}.")

    for epoch in range(start_epoch, epochs):
        df_shuffled = train_df.sample(frac=1).reset_index(drop=True)
        epoch_loss = 0.0
        num_batches = 0

        for start_idx in range(0, len(train_df), batch_size):
            end_idx = min(start_idx + batch_size, len(train_df))
            batch_df = df_shuffled.iloc[start_idx:end_idx]
            if len(batch_df) < 2:
                continue

            X_v_batch, text_query_batch, text_tokens_batch = [], [], []
            for _, row in batch_df.iterrows():
                X_v_batch.append(row[feature_col])
                text_query_batch.append([vocab[w] for w in ["describe", "action"]])
                text_tokens_batch.append([vocab[w] for w in text_mapping[int(row["target_class"])]])

            X_v_batch = np.array(X_v_batch).reshape(len(batch_df), 5, 16, 16, 3)
            text_query_batch = np.array(text_query_batch)
            text_tokens_batch = np.array(text_tokens_batch)

            model.S_v = model.vision_encoder.forward(X_v_batch)
            model.S_y = model.text_encoder.encode(text_tokens_batch)
            model.S_q_seq = model.text_encoder.get_sequence_embeddings(text_query_batch)

            model.S_y_hat = model.predictor.forward(model.S_v, model.S_q_seq)
            loss = model.criterion.forward(model.S_y_hat, model.S_y)

            model.backward()

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} | InfoNCE Loss: {avg_loss:.4f}")

        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
            save_training_checkpoint(
                model,
                text_mapping,
                vocab,
                epoch + 1,
                loss_history,
                checkpoint_path=checkpoint_path,
            )

    correct = sum(
        1
        for _, row in test_df.iterrows()
        if predict(model, row[feature_col], num_classes, text_mapping, vocab)
        == int(row["target_class"])
    )
    print(f"\nTrue (Unseen) Accuracy: {(correct / len(test_df)) * 100:.2f}%")

    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color="green")
    plt.title("VL-JEPA (Spatio-Temporal) Training Loss")
    plt.grid(True)
    plt.show()

    save_numpy_model(model, text_mapping, vocab, save_path="Playground/kth_numpy_model_final.npz")
    return model, text_mapping, vocab, loss_history


# Usage:
# trained_model, text_mapping, vocab, loss_history = run_training_and_plot(
#     "kth_temporal_features.csv",
#     epochs=500,
#     batch_size=4,
#     checkpoint_path="Playground/kth_training_checkpoint.npz",
#     checkpoint_every=5,
#     resume=True,
# )
