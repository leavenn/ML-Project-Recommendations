import numpy as np
import tensorflow as tf

def get_embeddings(model, X_scaled):
    embedding_model = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[6].output)
    embeddings = embedding_model.predict(X_scaled, batch_size=32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / norms
    return embeddings, emb_normed
