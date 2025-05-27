from collections import defaultdict

def add_to_cart(product_name, cart_counts, dataFrame):
    """
    Adds a product to the cart. If the product is already in the cart, increase its quantity.

    Parameters:
        product_name (str): Name of the product to add.
        cart_counts (dict): Dictionary mapping product indices to quantities in the cart.
        dataFrame (DataFrame): The full product data including product names.
    """
    # Find the index of the product by name
    idx = dataFrame.index[dataFrame['Product_Name'] == product_name][0]

    # Increment the quantity in the cart (or set to 1 if not present)
    cart_counts[idx] = cart_counts.get(idx, 0) + 1

def get_cart_category_weights(cart_counts, index_to_category):
    """
    Calculates normalized weights for each product category based on cart contents.

    Parameters:
        cart_counts (dict): Dictionary of cart contents (product index -> quantity).
        index_to_category (array-like): Mapping from product index to category.

    Returns:
        dict: Mapping from category name to its relative weight in the cart.
    """
    category_counts = defaultdict(int)
    total = 0

    # Count the total quantity per category
    for idx, qty in cart_counts.items():
        category = index_to_category[idx]
        category_counts[category] += qty
        total += qty

    # Normalize by total quantity to get weights
    return {k: v / total for k, v in category_counts.items()}

def print_cart(cart_counts, dataFrame):
    """
    Prints the current contents of the cart in a nicely formatted way.

    Parameters:
        cart_counts (dict): Dictionary of cart contents (product index -> quantity).
        dataFrame (DataFrame): The full product data including product names.
    """
    if not cart_counts:
        print("\nYour cart is empty.\n")
        return
    
    print("\n🛒 Your cart contains:")
    print("-" * 40)
    total_items = 0
    for i, (idx, qty) in enumerate(cart_counts.items(), start=1):
        product_name = dataFrame.loc[idx, 'Product_Name']
        print(f"{i:2}. {product_name:<30} x {qty}")
        total_items += qty
    print("-" * 40)
    print(f"Total items in cart: {total_items}\n")


