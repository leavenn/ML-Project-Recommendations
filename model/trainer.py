from data.loader import load_and_preprocess_data
from model.trainer import build_or_load_model
from model.recommender import recommend_from_cart
from cart.cart_logic import add_to_cart, print_cart

print("Loading app.py...")

# Number of product recommendations to return
NUMBER_OF_RECOMMENDATIONS = 5

# Path to saved model
MODEL_PATH = "model/model.keras"

# Load and preprocess the dataset
df_display, dataFrame, index_to_category, X_all_scaled, scaler = load_and_preprocess_data()

# Build or load the model and its embeddings
model, embeddings, emb_normed = build_or_load_model(X_all_scaled, dataFrame['Rating'].values, MODEL_PATH)

# Initialize cart dictionary to store selected products
cart_counts = {}

# Display list of all available products
print("Available products:")
for i, product in enumerate(dataFrame['Product_Name'].values, 1):
    print(f"{i}. {product}")

# Prepare lowercase version of product names for easier lookup
product_names_lower = [p.lower() for p in dataFrame['Product_Name'].values]

# Main loop for user interaction
while True:
    # Ask user to input product name
    chosen_name = input("Enter product to add to cart: ").strip()
    chosen_name_lower = chosen_name.lower()

    # Check if the product exists (case-insensitive)
    if chosen_name_lower in product_names_lower:
        # Retrieve original name with correct capitalization
        original_name = dataFrame['Product_Name'].values[product_names_lower.index(chosen_name_lower)]

        # Add product to cart
        add_to_cart(original_name, cart_counts, dataFrame)

        # Print current contents of the cart
        print_cart(cart_counts, dataFrame)
    else:
        # Notify user if the product doesn't exist
        print(f"Product '{chosen_name}' does not exist in the database. Please try again.\n")
        continue

    # Get product recommendations based on current cart contents
    recommended_indices = recommend_from_cart(
        embeddings,
        emb_normed,
        index_to_category,
        cart_counts,
        NUMBER_OF_RECOMMENDATIONS
    )

    # Display recommended products to the user
    print("Recommended products:")
    print(df_display.iloc[recommended_indices].to_string(index=False))
    print("\n")

