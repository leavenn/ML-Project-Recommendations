from data.loader import load_and_preprocess_data
from model.trainer import build_or_load_model
from model.recommender import recommend_from_cart
from cart.cart_logic import add_to_cart, print_cart

print("🚀 Loading app.py...")

# Number of recommendations to be generated
NUMBER_OF_RECOMMENDATIONS = 5

# Path where the model is stored or will be saved
MODEL_PATH = "model/model.keras"

# Load the dataset and preprocess features
df_display, dataFrame, index_to_category, X_all_scaled, scaler = load_and_preprocess_data()

# Build or load the model and get its embeddings
model, embeddings, emb_normed = build_or_load_model(X_all_scaled, dataFrame['Rating'].values, MODEL_PATH)

# Dictionary to keep track of cart contents (product index -> quantity)
cart_counts = {}

# Display all available products to the user
print("\n🛍️  Available products:")
for i, product in enumerate(dataFrame['Product_Name'].values, 1):
    print(f"{i:3}. {product}")

# Create a lowercase version of all product names to simplify lookup
product_names_lower = [p.lower() for p in dataFrame['Product_Name'].values]

# Infinite loop for user interaction
while True:
    # Prompt user for input
    chosen_name = input("\n🛒 Enter product to add to cart: ").strip()
    chosen_name_lower = chosen_name.lower()

    # If product exists in the dataset
    if chosen_name_lower in product_names_lower:
        # Find original product name (case-sensitive version)
        original_name = dataFrame['Product_Name'].values[product_names_lower.index(chosen_name_lower)]

        # Add product to the cart
        add_to_cart(original_name, cart_counts, dataFrame)

        # Print current contents of the cart
        print_cart(cart_counts, dataFrame)
    else:
        # Product not found, show error and prompt again
        print(f"❌ Product '{chosen_name}' does not exist in the database. Please try again.")
        continue  # Skip to next iteration

    # Generate product recommendations based on current cart
    recommended_indices = recommend_from_cart(
        embeddings,
        emb_normed,
        index_to_category,
        cart_counts,
        NUMBER_OF_RECOMMENDATIONS
    )

    # Display recommended products nicely
    print("\n✨ Recommended products:")
    print(df_display.iloc[recommended_indices].to_string(index=False))
    print("\n----------------------------------------")
