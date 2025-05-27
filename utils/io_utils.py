import os
import joblib
import tensorflow as tf

def save_full_model(model, embeddings, emb_normed, scaler, folder="model/"):
    os.makedirs(folder, exist_ok=True)
    tf.keras.models.save_model(model, os.path.join(folder, "model.keras"))
    joblib.dump(embeddings, os.path.join(folder, "embeddings.pkl"))
    joblib.dump(emb_normed, os.path.join(folder, "emb_normed.pkl"))
    joblib.dump(scaler, os.path.join(folder, "scaler.pkl"))
    print("Model, embeddings i scaler zapisane do folderu:", folder)

def load_full_model(folder="model/"):
    model = tf.keras.models.load_model(os.path.join(folder, "model.keras"))
    embeddings = joblib.load(os.path.join(folder, "embeddings.pkl"))
    emb_normed = joblib.load(os.path.join(folder, "emb_normed.pkl"))
    scaler = joblib.load(os.path.join(folder, "scaler.pkl"))
    print("Model, embeddings i scaler załadowane z folderu:", folder)
    return model, embeddings, emb_normed, scaler

    