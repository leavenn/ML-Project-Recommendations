import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_product_indices_by_ids(df, product_ids, id_col="ID"):
    print(f"Szukanie produktów o ID: {product_ids} w kolumnie '{id_col}'")
    print("Dostępne ID w dataFrame:", df["ID"].tolist()[:10])  # pokaż tylko pierwsze 10
    print("Typ kolumny ID:", df["ID"].dtype)

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



