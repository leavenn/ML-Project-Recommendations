import os
import tensorflow as tf
from model.embedder import get_embeddings
from utils.io_utils import save_full_model, load_full_model

def build_or_load_model(X, y, model_path="model/model.keras"):
    embeddings_path = os.path.join("models", "embeddings.pkl")
    emb_normed_path = os.path.join("model", "emb_normed.pkl")
    scaler_path = "model/scaler.pkl"

    if all(os.path.exists(p) for p in [model_path, embeddings_path, emb_normed_path, scaler_path]):
        model, embeddings, emb_normed, scaler = load_full_model(folder="model")
        print("Załadowano model i embeddings ze ścieżek")
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(
            X, y,
            validation_split=0.2,
            epochs=20,
            batch_size=8,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
            ]
        )
        embeddings, emb_normed = get_embeddings(model, X)

        scaler = None  # lub rzeczywisty obiekt, jeśli masz

        save_full_model(model, embeddings, emb_normed, scaler, folder="model")
        print("Model i embeddings wytrenowane i zapisane")

    return model, embeddings, emb_normed





