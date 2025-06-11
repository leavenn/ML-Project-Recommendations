import json
import os
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from data.loader import load_and_preprocess_data
from model.trainer import build_or_load_autoencoder_model
from model.recommender import recommend_similar_products
from cart.cart_logic import add_to_cart_by_id, remove_from_cart_by_id, update_cart_item_quantity

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Load content from the new JSON file
with open(config["data_paths"]["content_file"], "r") as f:
    content = json.load(f)
    category_descriptions = content.get("category_descriptions", {})

data_paths = config["data_paths"]
data_params = config["data_params"]
model_paths = config["model_paths"]
number_of_recommendations = config["number_of_recommendations"]

dataFrame, X, scaler = load_and_preprocess_data(data_paths, data_params)
if 'ID' not in dataFrame.columns:
    dataFrame['ID'] = dataFrame.index

# Add a mock 'Reviews' column if it doesn't exist for demonstration
if 'Reviews' not in dataFrame.columns:
    # Use a reproducible random state for consistency across reloads
    np.random.seed(42)
    # Generate a plausible number of reviews based on the product's rating
    dataFrame['Reviews'] = (dataFrame['Rating'].astype(float) * np.random.uniform(20, 80, size=len(dataFrame)) + np.random.randint(5, 50, size=len(dataFrame))).astype(int)

# Build or load the model and embeddings
autoencoder, embeddings, emb_normed = build_or_load_autoencoder_model(X, model_paths)

app = Flask(__name__)
app.secret_key = 'your_strong_random_secret_key_here_for_sessions'


# --- GLOBAL TEMPLATE VARIABLES ---
@app.context_processor
def inject_global_vars():
    store_name = 'Aura Beauty'
    return {
        'store_name': store_name,
        'logo_svg': f'<img src="/static/logo.svg" alt="{store_name} Logo" style="height: 100px; width: auto;">',
        'category_descriptions': category_descriptions
    }

unique_categories = sorted(dataFrame[data_params["columns_to_load"]["category_col"]].unique().tolist())

def sanitize_cart_keys(cart_dict):
    return {int(k): v for k, v in cart_dict.items() if str(k).isdigit()}

def get_current_cart_details():
    user_cart_counts = sanitize_cart_keys(session.get('cart_counts', {}))
    cart_details = []
    total_items = 0
    total_price = 0.0
    for idx, qty in user_cart_counts.items():
        if idx not in dataFrame.index: continue
        row = dataFrame.loc[idx]
        price = round(float(row['Price_USD']), 2)
        subtotal = round(price * qty, 2)
        cart_details.append({
            "id": int(idx), "product_name": row['Product_Name'], "quantity": qty,
            "price_per_item": price, "subtotal": subtotal, "category": row['Category']
        })
        total_items += qty
        total_price += subtotal
    return {"cart_items": cart_details, "total_items": total_items, "total_price": round(total_price, 2)}

def add_brand_and_category_to_products(products):
    for p in products:
        if p.get('ID') in dataFrame.index:
            product_data = dataFrame.loc[p['ID']]
            if 'Category' not in p:
                p['Category'] = product_data['Category']
            if 'Brand' not in p:
                p['Brand'] = product_data['Brand']
    return products

@app.route('/')
def home():
    cart = get_current_cart_details()
    cart_ids = [item['id'] for item in cart['cart_items']]
    page_data = {"recommendations": None, "categorized_products": None}

    if cart_ids:
        rec_df = recommend_similar_products(dataFrame, emb_normed, cart_ids, id_col='ID', top_n=8)
        page_data["recommendations"] = add_brand_and_category_to_products(rec_df.to_dict(orient='records'))
    else:
        categorized_products = {}
        for cat in unique_categories:
            category_df = dataFrame[dataFrame['Category'] == cat].sort_values(by='Rating', ascending=False)
            top_products = category_df.head(4).to_dict(orient='records')
            bestseller_ids = category_df.head(3)['ID'].tolist()
            for product in top_products: product['is_bestseller'] = product['ID'] in bestseller_ids
            categorized_products[cat] = add_brand_and_category_to_products(top_products)
        page_data["categorized_products"] = categorized_products
    return render_template('index.html', initial_cart=cart, categories=unique_categories, page_data=page_data)

@app.route('/product/<int:product_id>')
def product_detail(product_id):
    if product_id not in dataFrame.index: return "Product not found", 404
    product = dataFrame.loc[product_id].to_dict()
    product['ID'] = product_id
    product_category = product['Category']
    related_products_df = dataFrame[(dataFrame['Category'] == product_category) & (dataFrame.index != product_id)].sort_values(by='Rating', ascending=False).head(4)
    related_products = add_brand_and_category_to_products(related_products_df.to_dict(orient='records'))
    return render_template('product_detail.html', product=product, initial_cart=get_current_cart_details(), related_products=related_products, categories=unique_categories)

@app.route('/category/<string:category_name>')
def category_page(category_name):
    cart = get_current_cart_details()
    cart_ids = [item['id'] for item in cart['cart_items']]
    all_prods_df = dataFrame[dataFrame['Category'] == category_name].copy()
    bestseller_ids = all_prods_df.sort_values(by='Rating', ascending=False).head(3)['ID'].tolist()
    all_prods_df['is_bestseller'] = all_prods_df['ID'].isin(bestseller_ids)
    all_prods = all_prods_df.to_dict(orient='records')
    
    default_sort = "recommended" if cart_ids else "rating"
    sort = request.args.get("sort", default_sort)

    if sort == "recommended" and cart_ids:
        rec_df = recommend_similar_products(dataFrame, emb_normed, cart_ids, id_col='ID', top_n=500)
        all_rec_ids = rec_df['ID'].tolist()
        rec_id_set = set(all_rec_ids)
        
        recommended_in_category = []
        other_in_category = []
        
        for p in all_prods:
            if p['ID'] in rec_id_set:
                recommended_in_category.append(p)
            else:
                other_in_category.append(p)
                
        recommended_in_category.sort(key=lambda p: all_rec_ids.index(p['ID']))
        other_in_category.sort(key=lambda p: p['Rating'], reverse=True)
        
        sorted_prods = recommended_in_category + other_in_category
    elif sort == "price_asc":
        sorted_prods = sorted(all_prods, key=lambda p: p['Price_USD'])
    elif sort == "price_desc":
        sorted_prods = sorted(all_prods, key=lambda p: p['Price_USD'], reverse=True)
    else: # rating
        sorted_prods = sorted(all_prods, key=lambda p: p['Rating'], reverse=True)
        
    return render_template("category_page.html", category_name=category_name, products=sorted_prods, categories=unique_categories, cart=cart, current_sort=sort)

@app.route('/basket')
def basket_page():
    cart = get_current_cart_details()
    cart_ids = [item["id"] for item in cart["cart_items"]]
    if cart_ids:
        df = recommend_similar_products(dataFrame, emb_normed, cart_ids, id_col='ID', top_n=12)
        recs = add_brand_and_category_to_products(df.to_dict(orient='records'))
    else: recs = add_brand_and_category_to_products(dataFrame.sort_values(by='Rating', ascending=False).head(12).to_dict(orient='records'))
    return render_template("basket_page.html", cart=cart, related_products=recs, categories=unique_categories)

# --- API ROUTES ---
@app.route('/api/get_cart', methods=['GET'])
def api_get_cart():
    return jsonify(get_current_cart_details())

@app.route('/api/add_to_cart/<int:product_id>', methods=['POST'])
def api_add_to_cart(product_id):
    data = request.get_json() or {}
    quantity = data.get('quantity', 1)
    counts = sanitize_cart_keys(session.get('cart_counts', {}))
    add_to_cart_by_id(product_id, quantity, counts, dataFrame)
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

@app.route('/api/update_cart_quantity', methods=['POST'])
def api_update_cart_quantity():
    data = request.get_json()
    product_id, quantity = data.get('product_id'), data.get('quantity')
    counts = sanitize_cart_keys(session.get('cart_counts', {}))
    if update_cart_item_quantity(int(product_id), int(quantity), counts):
        session['cart_counts'], session.modified = counts, True
        return jsonify({"message": "Quantity updated.", **get_current_cart_details()})
    return jsonify({"error": "Failed to update quantity."}), 400

@app.route('/api/reset_cart', methods=['POST'])
def api_reset_cart():
    session['cart_counts'] = {}
    session.modified = True
    return jsonify({"message": "Basket reset.", **get_current_cart_details()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
