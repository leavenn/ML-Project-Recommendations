import numpy as np
import tensorflow as tf

def get_embeddings(model, X_scaled):
    # Create a sub-model that outputs the values from the penultimate layer (assumed to be embeddings)
    # model.layers[0].input: the input layer
    # model.layers[6].output: output of the 7th layer (zero-indexed)
    embedding_model = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[6].output)

    # Generate the embeddings for all input data
    embeddings = embedding_model.predict(X_scaled, batch_size=32)

    # Compute the L2 norm (Euclidean norm) of each embedding vector
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Normalize embeddings to have unit length (important for cosine similarity)
    emb_normed = embeddings / norms

    # Return both raw and normalized embeddings
    return embeddings, emb_normed

