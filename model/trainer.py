import os
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from model.embedder import get_embeddings
from utils.io_utils import save_full_model, load_full_model

def plot_history(history_data):
    plt.figure(figsize=(8, 5))
    plt.plot(history_data['loss'], label='Train Loss')
    if 'val_loss' in history_data:
        plt.plot(history_data['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def build_or_load_model(X, y, model_path="model/model.keras"):
    folder = "model"
    embeddings_path = os.path.join(folder, "embeddings.pkl")
    emb_normed_path = os.path.join(folder, "emb_normed.pkl")
    scaler_path = os.path.join(folder, "scaler.pkl")
    history_path = os.path.join(folder, "history.pkl")

    if all(os.path.exists(p) for p in [model_path, embeddings_path, emb_normed_path, scaler_path]):
        model, embeddings, emb_normed, scaler = load_full_model(folder=folder)
        print("Załadowano model i embeddings ze ścieżek")

        # Pokaż wykres z zapisanej historii, jeśli istnieje
        if os.path.exists(history_path):
            history_data = joblib.load(history_path)
            print("Pokazuję zapisany wykres strat...")
            plot_history(history_data)
        else:
            print("Nie znaleziono pliku z historią. Tworzę pusty wykres.")
            plot_history({'loss': [], 'val_loss': []})

    else:
        print("Trening nowego modelu...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        history = model.fit(
            X, y,
            validation_split=0.2,
            epochs=20,
            batch_size=8,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
            ]
        )

        # Zapis historii do pliku
        joblib.dump(history.history, history_path)

        # Wykres na świeżo
        plot_history(history.history)

        embeddings, emb_normed = get_embeddings(model, X)
        scaler = None  # Podmień jeśli używasz scalera

        save_full_model(model, embeddings, emb_normed, scaler, folder=folder)
        print("Model i embeddings wytrenowane i zapisane")

    return model, embeddings, emb_normed
