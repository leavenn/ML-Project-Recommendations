import numpy as np
from cart.cart_logic import get_cart_category_weights

def recommend_from_cart(embeddings, normalized_embeddings, index_to_category, cart_counts, numberOfRecommendations):
    # If the cart is empty, return no recommendations
    if not cart_counts:
        return []

    in_cart = set(cart_counts)  # Indices of products already in cart
    category_weights = get_cart_category_weights(cart_counts, index_to_category)  # Get weights for each category
    final_recommendations = []

    # Precompute arrays for faster access
    indices = np.array(list(cart_counts.keys()))  # Product indices in the cart
    quantities = np.array(list(cart_counts.values()))  # Corresponding quantities
    categories = np.array([index_to_category[i] for i in indices])  # Categories of products in the cart

    # Loop through each product category in the cart
    for category, weight in category_weights.items():
        # Calculate how many products to recommend from this category
        recs = max(1, round(weight * numberOfRecommendations))

        # Filter cart products that belong to this category
        mask = categories == category
        cat_indices = indices[mask]
        cat_quantities = quantities[mask]

        if len(cat_indices) == 0:
            continue  # Skip if no items from this category

        # Compute the weighted average embedding for this category
        total_cat = cat_quantities.sum()
        vec = np.sum(embeddings[cat_indices] * cat_quantities[:, None], axis=0) / total_cat
        vec /= np.linalg.norm(vec)  # Normalize the vector

        # Compute cosine similarity between the category vector and all product embeddings
        sim = normalized_embeddings @ vec

        # Mask out products that are not in this category or already in the cart
        mask_sim = np.array([index_to_category[i] == category and i not in in_cart for i in range(len(sim))])
        sim = np.where(mask_sim, sim, -np.inf)  # Assign -inf similarity to masked items

        # Select top N most similar products for this category
        top = np.argpartition(sim, -recs)[-recs:]
        top_sorted = top[np.argsort(sim[top])[::-1]]

        for i in top_sorted:
            if i not in final_recommendations:
                final_recommendations.append(i)

    # If fewer recommendations were found than requested, fill in with global similarity
    if len(final_recommendations) < numberOfRecommendations:
        # Compute a global weighted embedding vector from all cart items
        total = quantities.sum()
        vec = np.sum(embeddings[indices] * quantities[:, None], axis=0) / total
        vec /= np.linalg.norm(vec)

        sim = normalized_embeddings @ vec
        excluded = in_cart.union(final_recommendations)  # Avoid duplicates and cart items
        sim[list(excluded)] = -np.inf

        # Get additional recommendations to meet the required count
        filler_count = numberOfRecommendations - len(final_recommendations)
        filler = np.argpartition(sim, -filler_count)[-filler_count:]
        filler_sorted = filler[np.argsort(sim[filler])[::-1]]
        final_recommendations.extend(filler_sorted)

    return final_recommendations[:numberOfRecommendations]


