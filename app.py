import json
from data.loader import load_and_preprocess_data
from model.trainer import build_or_load_autoencoder_model
from model.recommender import plot_similar_products, recommend_similar_products
from cart.cart_logic import add_to_cart_by_id, print_cart

print("🚀 Loading app.py...")

# Wczytaj plik konfiguracyjny
with open("config.json", "r") as f:
    config = json.load(f)

# Rozdziel na osobne słowniki
data_paths = config["data_paths"]
data_params = config["data_params"]
model_paths = config["model_paths"]
output_paths = config["output_paths"]

# Number of recommendations to be generated
number_of_recommendations = config["number_of_recommendations"]

# Load the dataset and preprocess features
dataFrame, X, scaler = load_and_preprocess_data(data_paths, data_params)

# Build or load the model and get its embeddings
autoencoder, embeddings, emb_normed = build_or_load_autoencoder_model(X, model_paths)

cart_counts = {}

while True:
    try:
        chosen_id = int(input("\n🛒 Podaj ID produktu do dodania do koszyka: ").strip())
    except ValueError:
        print("❌ Podano nieprawidłowy ID (musi być liczbą całkowitą).")
        continue

    added = add_to_cart_by_id(chosen_id, cart_counts, dataFrame)
    if not added:
        continue

    print_cart(cart_counts, dataFrame)

    # Generowanie rekomendacji
    recommended_df = recommend_similar_products(
        df=dataFrame,
        embeddings_normed=emb_normed,
        product_ids=list(cart_counts.keys()),
        id_col=data_params["columns_to_load"]["product_id_col"],
        top_n=number_of_recommendations
    )

    print("\n✨ Proponowane produkty:")
    print(recommended_df[['Product_Name', 'Brand', 'Category', 'Price_USD', 'Rating', 'similarity']])
    
    # Rysuj wykres podobieństw
    plot_similar_products(recommended_df, title="🔎 Rekomendacje podobne do produktów w koszyku")

    

