import numpy as np
from cart.cart_logic import get_cart_category_weights

def recommend_from_cart(embeddings, normalized_embeddings, index_to_category, cart_counts, numberOfRecommendations):
    if not cart_counts:
        return []

    in_cart = set(cart_counts)
    category_weights = get_cart_category_weights(cart_counts, index_to_category)
    final_recommendations = []

    # Precompute arrays once for efficiency
    indices = np.array(list(cart_counts.keys()))
    quantities = np.array(list(cart_counts.values()))
    categories = np.array([index_to_category[i] for i in indices])

    for category, weight in category_weights.items():
        recs = max(1, round(weight * numberOfRecommendations))

        # Select items in cart from this category
        mask = categories == category
        cat_indices = indices[mask]
        cat_quantities = quantities[mask]

        if len(cat_indices) == 0:
            continue

        total_cat = cat_quantities.sum()
        # Compute weighted embedding vector for this category
        vec = np.sum(embeddings[cat_indices] * cat_quantities[:, None], axis=0) / total_cat
        vec /= np.linalg.norm(vec)

        # Compute similarity with normalized embeddings
        sim = normalized_embeddings @ vec

        # Mask out items not in this category or already in cart
        mask_sim = np.array([index_to_category[i] == category and i not in in_cart for i in range(len(sim))])
        sim = np.where(mask_sim, sim, -np.inf)

        # Get top recommendations for this category
        top = np.argpartition(sim, -recs)[-recs:]
        top_sorted = top[np.argsort(sim[top])[::-1]]
        for i in top_sorted:
            if i not in final_recommendations:
                final_recommendations.append(i)

    # If not enough recommendations, fill with global similarity excluding cart and already recommended
    if len(final_recommendations) < numberOfRecommendations:
        total = quantities.sum()
        vec = np.sum(embeddings[indices] * quantities[:, None], axis=0) / total
        vec /= np.linalg.norm(vec)

        sim = normalized_embeddings @ vec
        excluded = in_cart.union(final_recommendations)
        sim[list(excluded)] = -np.inf

        filler_count = numberOfRecommendations - len(final_recommendations)
        filler = np.argpartition(sim, -filler_count)[-filler_count:]
        filler_sorted = filler[np.argsort(sim[filler])[::-1]]
        final_recommendations.extend(filler_sorted)

    return final_recommendations[:numberOfRecommendations]

