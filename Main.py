import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter 

# !! Config
db_name = 'TestBase.csv' # beauty_cosmetics_products.csv
numberOfRecommendations = 5
print_graph = False;


# -- Cart
cart_indices=[]

def add_to_cart(product_name):
    # Find index of product_name in dataFrame and append to cart_indices.
    # TODO: change to multiple iterations of the same product, make it hodl weigth - more products from one cat -> more recomendations of that category
    idx = dataFrame.index[dataFrame['Product_Name']==product_name][0]
    if idx not in cart_indices:
        cart_indices.append(idx)

def get_cart_category_weigths(cart_indices, index_to_category):
    print("Log -0")
    # Returns a dictionary with category/weigth pairs
    counts = Counter(index_to_category[idx] for idx in cart_indices)
    total = sum(counts.values())
    print ("Log -1")
    # {} makes new dictionary with category as key and count/total as value
    # for each cat (category) make a value of count / total
    # This will give us the percentage of each category in the cart
    # normalize to sum = 1.0
    return {cat: cnt/total for cat, cnt in counts.items()}

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
X_train,X_val, y_train,y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# -- Scaling
# Standard Scaler - standarizes features to an average of 0 and a standard deviation of 1
scaler = StandardScaler()
# Fit the scaler to the training data
X_train = scaler.fit_transform(X_train)
# Transform the validation data using the fitted scaler
# This ensures that the validation data is scaled in the same way as the training data
X_val = scaler.transform(X_val)




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
# Layer 2
model.add(keras.layers.Dense(16, activation='relu'))
# Going from 32 to 16 neurons means comressing the data, which leads to isolating the most important features from 31 space

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
    epochs=20,
    # batch size - number of data per iteration
    # It will be trained on 8 samples at a time
    batch_size=8,
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


def recommend_from_cart(embeddings, emb_normed, index_to_category, cart_indices, numberOfRecommendations):
    print("Log 1")
    if not cart_indices:
        return None # if cart empty, return None
    
    # Get the category weights from the cart
    cat_weigths = get_cart_category_weigths(cart_indices, index_to_category)

    selected = []
    already = set(cart_indices)
    print("Log 2")
    for cat, weight in cat_weigths.items():
        print("Log 2.1")
        # how many recommendations for this category
        #The more products of the same category i nthe basket -> the more recommendations of that category
        m = max(1, round(weight * numberOfRecommendations))

        # Get the indices of the products in the category from the cart
        idxs_C = [i for i in cart_indices if index_to_category[i] == cat]

        # Normalize
        vec = embeddings[idxs_C].mean(axis=0)
        vec = vec / np.linalg.norm(vec)
        print("Log 2.2")
        # Cosine similarity
        # Calculate similarity between vec and all product embeddings
        # The dot product of two vectors is a measure of their similarity
        # produces a 1d array of vec compared to the embedding of the product
        sims = np.dot(vec, emb_normed.T)

        # mask - exclude everything not in the category or already in the cart
        print("Log 2.3")
        mask = [(index_to_category[i] == cat and i not in already) for i in range(len(sims))]
        # # set sims where mask==False to -inf setting their similarity to -inf
        sims = np.where(mask, sims, -np.inf)
        print("Log 2.4")
        # pick top k
        top_m = np.argsort(sims)[-m:][::-1]
        for idx in top_m:
            if idx not in selected:
                selected.append(idx)
    print("Log 3")
    # if we have less than numberOfRecommendations, fill the rest with global
    #TODO: change it to similar categories - home to food, beauty to home, so on.
    if(len(selected)<numberOfRecommendations):
        
        global_sims = embeddings[cart_indices].mean(axis=0)
        global_sims = global_sims / np.linalg.norm(global_sims)
        sims_global = np.dot(global_sims, emb_normed.T)
        needed = numberOfRecommendations - len(selected)
        fill = np.argsort(sims_global)[-needed:][::-1]
        selected.extend(fill.tolist())
    print("Log 4")
    return selected[:numberOfRecommendations]


    # cart_vector = embeddings[cart_indices].mean(axis=0)
    # # Normalize cart_vector
    # cart_vector = cart_vector / np.linalg.norm(cart_vector)
    # # Calculate similarity between cart_vector and all product embeddings
    # sims = np.dot(cart_vector, emb_normed.T)
   
    # #Determine categories present in cart
    # cart_cats = set(index_to_category[cart_indices])

    # # Exclude out everything not in those categories or already in cart
    # mask = np.array([ (idx not in cart_indices) and (index_to_category[idx] in cart_cats) for idx in range(len(sims))])

    # # set sims where mask==False to -inf setting their similarity to -inf
    # sims[~mask] = -np.inf

    # #Sort ascending and get the last 5, ::-1 - swap their order (from 1,2,3 to 3,2,1) so the most recomended is printed first
    # top_k = np.argsort(sims)[-numberOfRecommendations:][::-1]
    # return top_k


while True:
    chosen_name = input("Enter the name of the product you want to add to your cart: ")
    if chosen_name in dataFrame['Product_Name'].values:
        add_to_cart(chosen_name)
        print(f"{chosen_name} added to cart.")
        #Cart content 
        print("Cart content:")
        for idx in cart_indices:
            print(dataFrame.iloc[idx]['Product_Name'])
        print("\n")
    rec_idx = recommend_from_cart(
        embeddings, 
        emb_normed,
        index_to_category,
        cart_indices, 
        numberOfRecommendations
        )
    recs = df_display.loc[rec_idx]
    print(f"Recommendations for {chosen_name}:\n",recs.to_string(index=False))
    print("\n\n")
