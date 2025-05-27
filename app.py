from data.loader import load_and_preprocess_data
from model.trainer import build_or_load_model
from model.recommender import recommend_from_cart
from cart.cart_logic import add_to_cart, print_cart

print("Ładowanie app.py")
NUMBER_OF_RECOMMENDATIONS = 5
MODEL_PATH = "model/model.keras"

# Load & preprocess
df_display, dataFrame, index_to_category, X_all_scaled, scaler = load_and_preprocess_data()

# Model & embeddings
model, embeddings, emb_normed = build_or_load_model(X_all_scaled, dataFrame['Rating'].values, MODEL_PATH)
cart_counts = {}

print("Dostępne produkty:")
for i, product in enumerate(dataFrame['Product_Name'].values, 1):
    print(f"{i}. {product}")


# Przygotuj listę nazw produktów w lowercase do łatwiejszego sprawdzania
product_names_lower = [p.lower() for p in dataFrame['Product_Name'].values]

# Main loop
while True:
    chosen_name = input("Enter product to add to cart: ").strip()
    chosen_name_lower = chosen_name.lower()
    if chosen_name_lower in product_names_lower:
        # Znajdź oryginalną nazwę produktu z zachowaniem wielkości liter
        original_name = dataFrame['Product_Name'].values[product_names_lower.index(chosen_name_lower)]
        add_to_cart(original_name, cart_counts, dataFrame)
        print_cart(cart_counts, dataFrame)
    else:
        print(f"Produkt '{chosen_name}' nie istnieje w bazie. Spróbuj ponownie.")
        print("\n")
        continue
    
    recommended_indices = recommend_from_cart(embeddings, emb_normed, index_to_category, cart_counts, NUMBER_OF_RECOMMENDATIONS)
    print("Recommended products:")
    print(df_display.iloc[recommended_indices].to_string(index=False))
    print("\n")

