from collections import defaultdict

def add_to_cart(product_name, cart_counts, dataFrame):
    idx = dataFrame.index[dataFrame['Product_Name'] == product_name][0]
    cart_counts[idx] = cart_counts.get(idx, 0) + 1

def get_cart_category_weights(cart_counts, index_to_category):
    category_counts = defaultdict(int)
    total = 0
    for idx, qty in cart_counts.items():
        category = index_to_category[idx]
        category_counts[category] += qty
        total += qty
    return {k: v / total for k, v in category_counts.items()}

def print_cart(cart_counts, dataFrame):
    print("Cart:")
    for idx, qty in cart_counts.items():
        print(f"> {dataFrame.loc[idx, 'Product_Name']} - {qty}")

