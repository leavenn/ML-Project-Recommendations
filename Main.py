import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

# !! Config
db_name = 'beauty_cosmetics_products.csv' 
# TestBase.csv
# beauty_cosmetics_products.csv
number_of_recommendations = 5
print_graph = True;
test_size = 0.2
model_epochs = 20
save_model = True
load_model_from_file = True
saved_model_path = 'model.h5'
# -- End of Config


    


# -- Cart

# map of product/how many in the cart
cart_counts = {}

def add_to_cart(product_name):
    # Find index of product_name in dataFrame and append to cart_indices.
    product_idx = dataFrame.index[dataFrame['Product_Name']==product_name][0]
    cart_counts[product_idx] = cart_counts.get(product_idx, 0) + 1

def get_cart_category_weights(cart_counts, index_to_category):
    # if the values are nto set in the dictionary, set a default value of 0
    category_counts = defaultdict(int)
    total_count = 0

    # Iterate each product in the cart
    # for each product in the cart, get the category and add the quantity to the category count
    for product_idx, quantity in cart_counts.items():
        category = index_to_category[product_idx]
        category_counts[category] += quantity
        total_count += quantity
    
    # {} makes new dictionary with category as key and count/total as value
    # for each cat (category) make a value of count / total
    # This will give us the percentage of each category in the cart
    # normalize to sum = 1.0
    return {category: count/total_count for category, count in category_counts.items()}

# Read data from CSV to dataframe
dataFrame = pd.read_csv(db_name)
df_display = dataFrame[['Product_Name', 'Brand', 'Category', 'Price_USD', 'Rating']].copy()

# build an array of original categories, one per product index
index_to_category = df_display['Category'].values



# -- Dataset Normalization

# Create a new numeric column 'Size_ml' by stripping 'ml' and casting to int
dataFrame['Size_ml'] = dataFrame['Product_Size'].str.replace('ml', '').astype(int)
# Drop original text column once converted
dataFrame.drop(columns=['Product_Size'], inplace=True)

#Encode
# Convert categorical columns to numeric using one-hot encoding
freq_map = {'Occasional':1,'Monthly':2,'Weekly':3,'Daily':4}
dataFrame['Usage_Num'] = dataFrame['Usage_Frequency'].map(freq_map)
dataFrame.drop(columns=['Usage_Frequency'], inplace=True)

# Columns to one-hot encode
categories = ['Brand','Category','Skin_Type','Gender_Target','Packaging_Type',
        'Main_Ingredient','Country_of_Origin']
# drop first - first collumn gets dropped 
dataFrame = pd.get_dummies(dataFrame, columns=categories, drop_first=True)

# Binarization 'cruelty free' (T/F) to 0-1
dataFrame['Cruelty_Free'] = dataFrame['Cruelty_Free'].astype(int)

# -- End of Normalization

# -- Dataset Splitting
#y -  the target (what we want to predict), 1d Array
y = dataFrame['Rating'].values


# X - the dataset, 2d NumPy array
drop_cols = ['Product_Name', 'Rating']
X = dataFrame.drop(columns = drop_cols).values


# Split the dataset into training and testing sets
#random_state - seed for random number generator, ensures reproducibility
X_train,X_val, y_train,y_val = train_test_split(X, y, test_size = test_size, random_state=42)


# -- Scaling
# Standard Scaler - standarizes features to an average of 0 and a standard deviation of 1
scaler = StandardScaler()
# Fit the scaler to the training data
X_train = scaler.fit_transform(X_train)
# Transform the validation data using the fitted scaler
# This ensures that the validation data is scaled in the same way as the training data
X_val = scaler.transform(X_val)



# load model from file if it exists
if os.path.exists(saved_model_path) and load_model_from_file:
    #skip trainign if the model exist in file
    model = load_model(saved_model_path)
    print("Model loaded from file")
else:
    #-- Model
    # Blank sequential model
    # A sequential model is a linear stack of layers where each layer has exactly one input tensor and one output tensor.
    model = keras.Sequential()

    # Add a layer to the model
    # Dense Layer - each neuron is connected to every neuron in the previous layer 
    # Layer 1
    model.add(keras.layers.Dense(
        # number of neurons in the layer
        units = 32, 
        # activation function (Nonlinear)
        # Relu - Rectified Linear Unit
        #   f(x)=max(0,x) 
        #   If the input (x) is POSITIVE, the output is the input itself (x).
        #   If the input (x) is NEGATVE or ZERO, the output is zero.
        activation='relu', 
        # how many inputs per example
        # Describes how many features each example in dataset has
        input_shape=(X.shape[1],)
        )
    )

    # Dropout Layer - randomly sets a fraction of input units to 0 at each update during training time, which helps prevent overfitting
    model.add(keras.layers.Dropout(0.3))

    # Layer 2
    model.add(keras.layers.Dense(16, activation='relu'))
    # Going from 32 to 16 neurons means comressing the data, which leads to isolating the most important features from 31 space

    #Dropout layer
    model.add(keras.layers.Dropout(0.3))

    # Layer 3
    model.add(keras.layers.Dense(8, activation='relu'))

    #Dropout layer
    model.add(keras.layers.Dropout(0.3))

    # Layer 4
    # Output Layer
    model.add(
        keras.layers.Dense(
            # One neuron in the output layer
            # because we are predicting a single value (rating)
            units=1,
            # no nonlinearity for regression, meaning its gonna turn itno a weighted sum of the imputs
            #Typically used in the last layer of a regression model to get a single output value
            activation='linear'
        )
    )

    # Model learning
    model.compile(
        # Optimizer - updating the weights of the model during training based on the gradient of the loss function
        # Adam is an optymalization algorithm
        optimizer='adam',
        # Loss function - measures how well the model is performing
        # Mean Squared Error (MSE) - measures the average of the squares of the errors
        loss='mean_squared_error',
        # Metrics - used to evaluate the performance of the model
        # Mean Absolute Error (MAE) - measures the average of the absolute differences between predicted and actual values
        metrics=['mean_absolute_error']
    )

    model.build(input_shape=(None, X.shape[1]))

    # Early stopping - stops training when the validation loss stops improving
    # This helps to prevent overfitting (earlier reading were at 1.7 loss)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3, # number of epochs with no improvement after which training will be stopped
        restore_best_weights=True, # restore model weights from the epoch with the best value of the monitored quantity
        )
    # Reduce learning rate when a metric has stopped improving
    # This reduces the learning rate when the validation loss stops improving
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5, # factor by which the learning rate will be reduced
        patience=3, # number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-6, # minimum learning rate
    )
    # --Model learning
    history = model.fit(
        # X_train - training data
        X_train,
        # y_train - training labels
        y_train,
        # Validation data - used to evaluate the model during training
        # Evaluates the model on the validation data at the end of each epoch
        # helps with spotting overtraining
        validation_data=(X_val, y_val),
        # number of epochs - how many times to loop over the whole dataset
        epochs = model_epochs,
        # batch size - number of data per iteration
        # It will be trained on 8 samples at a time
        batch_size=8,
        #Early stopping callback
        # This will stop the training if the validation loss does not improve for 3 epochs
        callbacks= [early_stop, reduce_lr],
    )

    # Graph
    if print_graph:
        plt.plot(history.history['loss'],label = 'train loss')
        plt.plot(history.history['val_loss'],label = 'val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.show()
        
    # --End of model training


# -- Save the model so it doesnt have to re-train
if save_model:
    model.save(saved_model_path)






# --End of model training

#Scale the entire dataset minus name, rating
# This is done to ensure that the model can be used for predictions on new data
X_all = dataFrame.drop(columns=['Product_Name', 'Rating']).values
X_all_scaled = scaler.transform(X_all)

# Get the embeddings from the model
# Embedding - a lower-dimensional vector representation of the input data
# Define the function to bring up the values from the 1st layer (model.layers[1]) - Layer counting from 0 
embedding_model = tf.keras.Model(
    inputs=model.layers[0].input,
    outputs=model.layers[1].output
)
embeddings = embedding_model.predict(X_all_scaled, batch_size=32)
# Normalize the embeddings
# This is done to ensure that the embeddings are on the same scale
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
# Embedings are divided by their norms
# Normalization of the embedding vector 
emb_normed = embeddings / norms

# Calculate the similarity matrix
# This is done to find the similarity between the embeddings
# its a 2d array where each element is the dot product of the corresponding embeddings
# The dot product of two vectors is a measure of their similarity
#The element at position (i, j) in this matrix will contain a measure of similarity (cosine of angle) between the embedding of product i and the embedding of product j.
sim_matrix = np.dot(emb_normed, emb_normed.T)


def recommend_from_cart(embeddings, normalized_embeddings, index_to_category, cart_counts, numberOfRecommendations):
    if not cart_counts:
        return [] # if cart empty, return None
    # Get the category weights from the cart
    category_weights = get_cart_category_weights(cart_counts, index_to_category)

    final_recommendations = []
    items_already_in_cart = set(cart_counts)
    for category, weight in category_weights.items():

        # how many recommendations for this category
        #The more products of the same category i nthe basket -> the more recommendations of that category
        recs_for_category = max(1, round(weight * numberOfRecommendations))
        # Get the indices of the products in the category
        total_in_category = sum(quantity for index, quantity in cart_counts.items() if index_to_category[index] == category)
        # build weighted sum for this category
        category_weighted_sum = np.zeros(embeddings.shape[1])
        for product_idx, quantity in cart_counts.items():
            if index_to_category[product_idx] == category:
                category_weighted_sum += embeddings[product_idx] * quantity
        # Normalize the category weighted sum
        category_vector  = category_weighted_sum / total_in_category

        # Normalize the category vector
        category_vector = category_vector / np.linalg.norm(category_vector)
        # Cosine similarity
        # Calculate similarity between vec and all product embeddings
        # The dot product of two vectors is a measure of their similarity
        # produces a 1d array of vec compared to the embedding of the product
        similarity_scores = np.dot(category_vector , normalized_embeddings.T)
        # mask - exclude everything not in the category or already in the cart
        valid_mask = [(index_to_category[i] == category) and (i not in items_already_in_cart ) for i in range(len(similarity_scores))]
        # # set sims where mask==False to -inf setting their similarity to -inf
        similarity_scores = np.where(valid_mask, similarity_scores, -np.inf)
        # pick top k
        top_indices = np.argsort(similarity_scores)[-recs_for_category:][::-1]
        for recommended_idx in top_indices:
            if recommended_idx not in final_recommendations :
                final_recommendations .append(recommended_idx)
    # if we have less than numberOfRecommendations, fill the rest with global
    if(len(final_recommendations )<numberOfRecommendations):
        
        total_items = sum(cart_counts.values())
        weighted_sum_vector = np.zeros(embeddings.shape[1])
        for product_idx, quantity in cart_counts.items():
            weighted_sum_vector += embeddings[product_idx]*quantity
        overall_cart_vector = weighted_sum_vector / total_items
        # Normalize the overall cart vector
        # This ensures that the overall cart vector is on the same scale as the embeddings
        overall_cart_vector = overall_cart_vector / np.linalg.norm(overall_cart_vector)
        all_similarity = np.dot(overall_cart_vector, normalized_embeddings.T)
        # Exclude items already in cart or already recommended
        excluded = items_already_in_cart.union(final_recommendations)
        # Set the excluded values to -inf
        # This will ensure that these items are not recommended 
        for idx in excluded:
            all_similarity[idx] = -np.inf
        remaining = numberOfRecommendations - len(final_recommendations )
        filler_indices = np.argsort(all_similarity)[-remaining:][::-1]
        final_recommendations.extend(filler_indices.tolist())

    return final_recommendations [:numberOfRecommendations]


# -- Main loop
while True:
    chosen_name = input("Enter the name of the product you want to add to your cart: ")
    if chosen_name in dataFrame['Product_Name'].values:
        add_to_cart(chosen_name)
        print(f"{chosen_name} added to cart.")
        #Cart content 
        print("Cart content:")
        for product_index, quantity in cart_counts.items():
            product_name = dataFrame.loc[product_index, 'Product_Name']
            print(f"> {product_name} - {quantity}")
        print("\n")
    recommended_indices = recommend_from_cart(
        embeddings, 
        emb_normed,
        index_to_category,
        cart_counts, 
        number_of_recommendations
        )
    print (f"Recommended indices: {recommended_indices}")
    recommendations = df_display.iloc[recommended_indices]
    print(f"Recommendations for {chosen_name}:\n",recommendations.to_string(index=False))
    print("\n\n")
