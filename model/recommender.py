from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def plot_similar_products(recommended_df, title="Top podobne produkty"):
    if recommended_df.empty:
        print("Brak danych do wykresu.")
        return

    names = recommended_df["Product_Name"] if "Product_Name" in recommended_df.columns else recommended_df.index.astype(str)
    scores = recommended_df["similarity"]

    plt.figure(figsize=(8, 5))
    plt.barh(names, scores, color='skyblue')
    plt.xlabel("Podobieństwo (cosine)")
    plt.title(title)
    plt.gca().invert_yaxis()  # Najbardziej podobny u góry
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def get_product_indices_by_ids(df, product_ids, id_col="ID"):

    if isinstance(product_ids, (int, str)):
        product_ids = [product_ids]
    indices = df[df[id_col].isin(product_ids)].index.tolist()
    if not indices:
        raise ValueError("Żaden z podanych ID nie został znaleziony.")
    return indices

def get_mean_embedding(embeddings_normed, indices):
    return np.mean(embeddings_normed[indices], axis=0).reshape(1, -1)

def recommend_similar_products(df, embeddings_normed, product_ids, id_col="ID", top_n=5):
    indices = get_product_indices_by_ids(df, product_ids, id_col)
    query_embedding = get_mean_embedding(embeddings_normed, indices)

    similarities = cosine_similarity(query_embedding, embeddings_normed).flatten()
    sorted_indices = np.argsort(-similarities)

    # Pomiń produkty wejściowe
    filtered_indices = [i for i in sorted_indices if i not in indices][:top_n]

    recommended_df = df.iloc[filtered_indices].copy()
    recommended_df["similarity"] = similarities[filtered_indices]
    return recommended_df



