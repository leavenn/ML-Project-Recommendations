import os
import joblib
import tensorflow as tf
from model.embedder import get_embeddings
from utils.io_utils import save_full_model, load_full_model

def build_or_load_autoencoder_model(X, model_paths):
    # Jeśli wszystkie pliki istnieją, załaduj model
    if all(os.path.exists(p) for p in model_paths.values() if p):
        load_full_model(model_paths)
        autoencoder, embeddings, emb_normed = load_full_model(model_paths)
        print("✅ Załadowano model i embeddings ze ścieżek.")
        return autoencoder, embeddings, emb_normed
    else:
        print("🛠 Trening nowego autoenkodera...")

        # Budowa architektury autoenkodera
        input_dim = X.shape[1]  # 4 cechy wejściowe
        encoding_dim = 8        # wymiar wewnętrzny (embedding)

        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoded = tf.keras.layers.Dense(32, activation='relu')(input_layer)
        encoded = tf.keras.layers.Dense(16, activation='relu')(encoded)
        embedding = tf.keras.layers.Dense(encoding_dim, activation='relu', name="embedding")(encoded)

        decoded = tf.keras.layers.Dense(16, activation='relu')(embedding)
        decoded = tf.keras.layers.Dense(32, activation='relu')(decoded)
        output_layer = tf.keras.layers.Dense(input_dim, activation='linear')(decoded)

        autoencoder = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')

        # Trening: wejście = wyjście (autoenkoder)
        history = autoencoder.fit(
            X, X,
            validation_split=0.2,
            epochs=20,
            batch_size=8,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)
            ]
        )

        # Zapisz historię treningu
        joblib.dump(history.history, model_paths['training_history'])

        # Wyciągnij embeddingi
        embeddings, emb_normed = get_embeddings(autoencoder, X)

        # Zapisz wszystko
        save_full_model(autoencoder, embeddings, emb_normed, model_paths)
        print("✅ Autoenkoder i embeddingi zostały zapisane.")

    return autoencoder, embeddings, emb_normed
