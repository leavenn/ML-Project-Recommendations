import json
import os
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from data.loader import load_and_preprocess_data
from model.trainer import build_or_load_autoencoder_model
from model.recommender import recommend_similar_products
from cart.cart_logic import add_to_cart_by_id, remove_from_cart_by_id, update_cart_item_quantity

# Load the application configuration from a JSON file.
# This keeps settings like file paths and model parameters separate from the code,
# making it easier to manage and change settings without altering the main logic.
with open("config.json", "r") as f:
    config = json.load(f)

# Load UI content - category descriptions, from a separate JSON file.
with open(config["data_paths"]["content_file"], "r") as f:
    content = json.load(f)
    category_descriptions = content.get("category_descriptions", {})

# Extract specific configuration sections for easier access throughout the app.
data_paths = config["data_paths"]
data_params = config["data_params"]
model_paths = config["model_paths"]
number_of_recommendations = config["number_of_recommendations"]

# Load and preprocess the product data using custom function.
# This returns the main DataFrame, the feature set (X) for the model, and the scaler used.
dataFrame, X, scaler = load_and_preprocess_data(data_paths, data_params)

# Build a new autoencoder model or load a pre-trained one from disk.
# This also generates the product "embeddings" - numerical representations of products that the model uses to understand similarity.
autoencoder, embeddings, emb_normed = build_or_load_autoencoder_model(
    X, model_paths)


# -- FLASK APP INITIALIZATION --
app = Flask(__name__)
# Set a secret key for session management. This is crucial for securely tracking user data like the contents of their shopping cart between requests.
app.secret_key = 'your_strong_random_secret_key_here_for_sessions'


# --- GLOBAL & HELPER FUNCTIONS ---

# --- GLOBAL TEMPLATE VARIABLES ---

# This function uses a context processor to make variables available in all Jinja2 templates.
# It's a clean way to provide global data (like the store name) without passing it in every `render_template` call.
@app.context_processor
def inject_global_vars():
    store_name = 'Aura Beauty'
    return {
        'store_name': store_name,
        'logo_svg': f'<img src="/static/logo.svg" alt="{store_name} Logo" style="height: 80px; width: auto;">',
        'category_descriptions': category_descriptions
    }


# Get a sorted list of unique product categories to display in the navigation menu.
unique_categories = sorted(
    dataFrame[data_params["columns_to_load"]["category_col"]].unique().tolist())


# Helper function to ensure cart keys stored in the session are integers.
# When a dictionary is stored in a session, its keys are converted to strings.
# We need to convert them back to integers to use them as product IDs.
def sanitize_cart_keys(cart_dict):
    return {int(k): v for k, v in cart_dict.items() if str(k).isdigit()}


# Fetches detailed information for items currently in the user's session cart.
# It calculates subtotals, total price, and total item count.
def get_current_cart_details():
    # Retrieve the cart from the session, sanitizing keys to ensure they are integers.
    user_cart_counts = sanitize_cart_keys(session.get('cart_counts', {}))
    cart_details = []
    total_items = 0
    total_price = 0.0

    # Loop through each product ID and its quantity in the cart.
    for idx, qty in user_cart_counts.items():
        # A safety check to ensure the product still exists in main dataframe.
        if idx not in dataFrame.index:
            continue
        row = dataFrame.loc[idx]
        price = round(float(row['Price_USD']), 2)
        subtotal = round(price * qty, 2)
        cart_details.append({
            "id": int(idx), "product_name": row['Product_Name'], "quantity": qty,
            "price_per_item": price, "subtotal": subtotal, "category": row['Category']
        })
        # Accumulate total items and price.
        total_items += qty
        total_price += subtotal
    # Return a dictionary containing the list of cart items and the overall totals.
    return {"cart_items": cart_details, "total_items": total_items, "total_price": round(total_price, 2)}

# Adds 'Brand' and 'Category' information to a list of product dictionaries.
# This is useful for enriching recommendation lists that might only contain basic product info.
# THIS IS JUST VISUAL, DOES NOT CHANGE THE DATAFRAME.
# Only displays info on the product tile/details page


def add_brand_and_category_to_products(products):
    for p in products:
        if p.get('ID') in dataFrame.index:
            product_data = dataFrame.loc[p['ID']]
            # Only add the data if it's not already present.
            if 'Category' not in p:
                p['Category'] = product_data['Category']
            if 'Brand' not in p:
                p['Brand'] = product_data['Brand']
    return products


#  --- PAGE ROUTES ---

# --- Home Page Route ---
@app.route('/')
def home():
    cart = get_current_cart_details()
    cart_ids = [item['id'] for item in cart['cart_items']]
    page_data = {"recommendations": None, "categorized_products": None}

    # Main Logic: Recommendations vs. Generic Content
    # If the user's cart has items, we generate personalized recommendations for them.
    # If the cart is empty, we show a generic view with top-rated products from each category.
    if cart_ids:
        # Generate recommendations based on the items in the cart.
        rec_df = recommend_similar_products(
            dataFrame, emb_normed, cart_ids, id_col='ID', top_n=8)
        page_data["recommendations"] = add_brand_and_category_to_products(
            rec_df.to_dict(orient='records'))
    else:
        # If the cart is empty, showcase top products from each category.
        categorized_products = {}
        for cat in unique_categories:
            # Filter products by category and sort by rating to find the bestsellers.
            category_df = dataFrame[dataFrame['Category'] == cat].sort_values(
                by='Rating', ascending=False)
            top_products = category_df.head(4).to_dict(orient='records')
            bestseller_ids = category_df.head(3)['ID'].tolist()
            # Tag the top 3 as "bestsellers" for the UI. Only visual.
            for product in top_products:
                product['is_bestseller'] = product['ID'] in bestseller_ids
            categorized_products[cat] = add_brand_and_category_to_products(
                top_products)
        page_data["categorized_products"] = categorized_products
    # Render the main index page with the prepared data.
    return render_template('index.html', initial_cart=cart, categories=unique_categories, page_data=page_data)

# --- Product Detail Page Route ---


@app.route('/product/<int:product_id>')
def product_detail(product_id):
    # Early stop - Handle cases where a product ID is invalid.
    if product_id not in dataFrame.index:
        return "Product not found", 404

    # Fetch all details for the requested product.
    product = dataFrame.loc[product_id].to_dict()
    product['ID'] = product_id
    product_category = product['Category']

    # Find related products by looking for other highly-rated items in the same category.
    related_products_df = dataFrame[(dataFrame['Category'] == product_category) & (
        dataFrame.index != product_id)].sort_values(by='Rating', ascending=False).head(4)
    related_products = add_brand_and_category_to_products(
        related_products_df.to_dict(orient='records'))

    return render_template('product_detail.html', product=product, initial_cart=get_current_cart_details(), related_products=related_products, categories=unique_categories)

# --- Category Page Route ---


@app.route('/category/<string:category_name>')
def category_page(category_name):
    cart = get_current_cart_details()
    cart_ids = [item['id'] for item in cart['cart_items']]

    # Get all products for the specified category and identify bestsellers.
    all_prods_df = dataFrame[dataFrame['Category'] == category_name].copy()
    bestseller_ids = all_prods_df.sort_values(
        by='Rating', ascending=False).head(3)['ID'].tolist()
    all_prods_df['is_bestseller'] = all_prods_df['ID'].isin(bestseller_ids)
    all_prods = all_prods_df.to_dict(orient='records')

    # The default sort is "recommended" if there are items in the cart
    # When the cart is empty - default sort is by "rating".
    default_sort = "recommended" if cart_ids else "rating"
    sort = request.args.get("sort", default_sort)

    if sort == "recommended" and cart_ids:
        # Get a large list of recommendations based on the user's cart.
        rec_df = recommend_similar_products(
            dataFrame, emb_normed, cart_ids, id_col='ID', top_n=500)
        all_rec_ids = rec_df['ID'].tolist()
        rec_id_set = set(all_rec_ids)

        recommended_in_category = []
        other_in_category = []

        # Split the category's products into two groups: recommended and not recommended.
        for p in all_prods:
            if p['ID'] in rec_id_set:
                recommended_in_category.append(p)
            else:
                other_in_category.append(p)

        # Sort the recommended products based on their recommendation rank.
        recommended_in_category.sort(key=lambda p: all_rec_ids.index(p['ID']))
        other_in_category.sort(key=lambda p: p['Rating'], reverse=True)

        # Combine the lists to show recommended items first.
        sorted_prods = recommended_in_category + other_in_category
    elif sort == "price_asc":
        sorted_prods = sorted(all_prods, key=lambda p: p['Price_USD'])
    elif sort == "price_desc":
        sorted_prods = sorted(
            all_prods, key=lambda p: p['Price_USD'], reverse=True)
    else:  # Default to sorting by rating.
        sorted_prods = sorted(
            all_prods, key=lambda p: p['Rating'], reverse=True)

    return render_template("category_page.html", category_name=category_name, products=sorted_prods, categories=unique_categories, cart=cart, current_sort=sort)


# --- Shopping Basket Page Route ---
@app.route('/basket')
def basket_page():
    cart = get_current_cart_details()
    cart_ids = [item["id"] for item in cart["cart_items"]]

    # Show related products on the basket page.
    # If the cart has items, recommendations are based on the cart.
    # If not, show top-rated products from the whole store.
    if cart_ids:
        df = recommend_similar_products(
            dataFrame, emb_normed, cart_ids, id_col='ID', top_n=12)
        recs = add_brand_and_category_to_products(df.to_dict(orient='records'))
    else:
        recs = add_brand_and_category_to_products(dataFrame.sort_values(
            by='Rating', ascending=False).head(12).to_dict(orient='records'))
    return render_template("basket_page.html", cart=cart, related_products=recs, categories=unique_categories)

# --- API ROUTES (for JavaScript interaction) ---

# API: Get Cart Details
# Provides the current cart state to the frontend JavaScript.


@app.route('/api/get_cart', methods=['GET'])
def api_get_cart():
    return jsonify(get_current_cart_details())

# API: Add to Cart
# Handles adding a product to the cart.


@app.route('/api/add_to_cart/<int:product_id>', methods=['POST'])
def api_add_to_cart(product_id):
    data = request.get_json() or {}
    quantity = data.get('quantity', 1)
    counts = sanitize_cart_keys(session.get('cart_counts', {}))
    add_to_cart_by_id(product_id, quantity, counts, dataFrame)
    # Save the updated cart back to the session.
    session['cart_counts'] = counts
    session.modified = True
    return jsonify({"message": "Product added.", **get_current_cart_details()})


@app.route('/api/remove_from_cart/<int:product_id>', methods=['POST'])
def api_remove_from_cart(product_id):
    counts = sanitize_cart_keys(session.get('cart_counts', {}))
    remove_from_cart_by_id(product_id, counts)
    session['cart_counts'] = counts
    session.modified = True
    return jsonify({"message": "Product removed.", **get_current_cart_details()})

# API: Remove from Cart
# Handles removing an entire item from the cart.


@app.route('/api/update_cart_quantity', methods=['POST'])
def api_update_cart_quantity():
    data = request.get_json()
    product_id, quantity = data.get('product_id'), data.get('quantity')
    counts = sanitize_cart_keys(session.get('cart_counts', {}))
    # The update function returns True on success.
    if update_cart_item_quantity(int(product_id), int(quantity), counts):
        session['cart_counts'], session.modified = counts, True
        return jsonify({"message": "Quantity updated.", **get_current_cart_details()})
    return jsonify({"error": "Failed to update quantity."}), 400

# API: Reset Cart
# Clears all items from the shopping cart.


@app.route('/api/reset_cart', methods=['POST'])
def api_reset_cart():
    session['cart_counts'] = {}
    session.modified = True
    return jsonify({"message": "Basket reset.", **get_current_cart_details()})


#  --- APPLICATION ENTRY POINT ---
if __name__ == '__main__':
    # Runs the Flask development server.
    # `debug=True` enables auto-reloading on code changes and provides detailed error pages.
    # `host='0.0.0.0'` makes the server accessible from other devices on the network.
    # by default the address is localhost:5000
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
