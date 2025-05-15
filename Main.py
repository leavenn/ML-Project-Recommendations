import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
    #SOPER
# Read data from CSV to dataframe
dataFrame = pd.read_csv('beauty_cosmetics_products.csv')



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
# --Model learning
history = model.fit(
    # X_train - training data
    X_train,
    # y_train - training labels
    y_train,
    # Validation data - used to evaluate the model during training
    # Evaluates the model on the validation data at the end of each epoch
    # helps with spotting overtraining
    validate_data=(X_val, y_val),
    # number of epochs - how many times to loop over the whole dataset
    epochs=50,
    # batch size - number of data per iteration
    # It will be trained on 8 samples at a time
    batch_size=8,
)



# Graph

plt.plot(history.history['loss'],label = 'train loss')
plt.plot(history.history['val_loss'],label = 'val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()