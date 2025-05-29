def add_to_cart_by_id(product_id, cart_counts, dataFrame):
    """
    Dodaje produkt do koszyka na podstawie jego ID.
    """
    if product_id not in dataFrame.index:
        print(f"❌ ID {product_id} nie istnieje w bazie danych.")
        return False
    cart_counts[product_id] = cart_counts.get(product_id, 0) + 1
    return True

def print_cart(cart_counts, dataFrame):
    """
    Wypisuje zawartość koszyka na podstawie ID produktów.
    """
    if not cart_counts:
        print("\n🛒 Koszyk jest pusty.\n")
        return

    print("\n🛒 Zawartość koszyka:")
    print("-" * 40)
    total_items = 0
    for i, (idx, qty) in enumerate(cart_counts.items(), start=1):
        row = dataFrame.loc[idx]
        print(f"{i:2}. {row['Product_Name']:<30} x {qty}")
        total_items += qty
    print("-" * 40)
    print(f"Liczba produktów w koszyku: {total_items}\n")



