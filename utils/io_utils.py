import os
import joblib
import tensorflow as tf

def save_full_model(autoencoder, embeddings, emb_normed, model_paths):
    tf.keras.models.save_model(autoencoder, model_paths['autoencoder_file'])
    joblib.dump(embeddings, model_paths['embeddings'])
    joblib.dump(emb_normed, model_paths['normalized_embeddings'])

def load_full_model(model_paths):
    model = tf.keras.models.load_model(model_paths['autoencoder_file'])
    embeddings = joblib.load(model_paths['embeddings'])
    emb_normed = joblib.load(model_paths['normalized_embeddings'])
    return model, embeddings, emb_normed

    