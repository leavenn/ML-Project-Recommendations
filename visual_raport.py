import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from utils.io_utils import load_full_model
from data.loader import load_and_preprocess_data

# Wczytaj konfigurację
with open("config.json", "r") as f:
    config = json.load(f)

data_paths = config["data_paths"]
data_params = config["data_params"]
model_paths = config["model_paths"]

# Załaduj model i dane
model, embeddings, emb_normed = load_full_model(model_paths)
dataFrame, X, _ = load_and_preprocess_data(data_paths, data_params)

# Oblicz błędy rekonstrukcji
reconstructed = model.predict(X)
mse_errors = np.mean((X - reconstructed) ** 2, axis=1)
dataFrame["Reconstruction_Error"] = mse_errors

# Top 10 największych błędów
top_errors = dataFrame.sort_values("Reconstruction_Error", ascending=False).head(10)

# Top 10 błędów
plt.figure(figsize=(10, 6))
sns.barplot(data=top_errors, y="Product_Name", x="Reconstruction_Error", palette="Reds_r", ci=None)
plt.title("Top 10 produktów z największym błędem rekonstrukcji")
plt.xlabel("Błąd")
plt.ylabel("Produkt")
plt.tight_layout()
plt.show()

# Cena vs Błąd
plt.figure(figsize=(8, 6))
sns.scatterplot(data=dataFrame, x="Price_USD", y="Reconstruction_Error", alpha=0.6)
plt.title("Cena vs Błąd rekonstrukcji")
plt.xlabel("Cena (USD)")
plt.ylabel("Błąd")
plt.grid(True)
plt.tight_layout()
plt.show()