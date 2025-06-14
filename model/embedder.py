import numpy as np
from sklearn.preprocessing import normalize
import tensorflow as tf


def get_embeddings(model, X):
    """
    Tworzy nowy model, który kończy się na warstwie 'embedding',
    a następnie generuje embeddingi dla X.
    Zwraca również wersję znormalizowaną (L2).
    """
    # Znajdź warstwę embedding po nazwie
    embedding_layer = model.get_layer("embedding")

    # Utwórz nowy model od wejścia do tej warstwy
    embedding_model = tf.keras.Model(
        inputs=model.input, outputs=embedding_layer.output)

    # Oblicz embeddingi
    embeddings = embedding_model.predict(X)

    # Normalizuj do długości 1 (L2 norm) – przydatne do porównań kosinusowych
    embeddings_normalized = normalize(embeddings)

    return embeddings, embeddings_normalized
