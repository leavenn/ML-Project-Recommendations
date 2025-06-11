def add_to_cart_by_id(product_id, quantity, cart_counts, dataFrame):
    """
    Adds a product to the cart based on its ID and quantity.
    This function was modified to accept a quantity parameter.
    """
    if product_id not in dataFrame.index:
        print(f"❌ ID {product_id} does not exist in the database.")
        return False
    # Add the specified quantity to the cart
    cart_counts[product_id] = cart_counts.get(product_id, 0) + quantity
    return True

def remove_from_cart_by_id(product_id, cart_counts):
    """
    Removes a product from the cart based on its ID.
    This now removes the entire product line from the cart.
    """
    if product_id in cart_counts:
        del cart_counts[product_id]
        return True
    else:
        print(f"❌ Product with ID {product_id} is not in the cart.")
        return False

def update_cart_item_quantity(product_id, quantity, cart_counts):
    """
    Updates the quantity of a specific item in the cart.
    """
    if product_id in cart_counts:
        if quantity > 0:
            cart_counts[product_id] = quantity
            return True
        else:
            # If the new quantity is 0 or less, remove the item entirely.
            del cart_counts[product_id]
            return True
    return False

def print_cart(cart_counts, dataFrame):
    """
    Prints the contents of the cart based on product IDs.
    """
    if not cart_counts:
        print("\n🛒 Cart is empty.\n")
        return

    print("\n🛒 Cart contents:")
    print("-" * 40)
    total_items = 0
    for i, (idx, qty) in enumerate(cart_counts.items(), start=1):
        if idx in dataFrame.index:
            row = dataFrame.loc[idx]
            print(f"{i:2}. {row['Product_Name']:<30} x {qty}")
            total_items += qty
    print("-" * 40)
    print(f"Total items in cart: {total_items}\n")
