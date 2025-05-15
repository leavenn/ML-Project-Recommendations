import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Read data from CSV to dataframe
dataFrame = pd.read_csv('data.csv')

# X - the dataset, 2d NumPy array
X = dataFrame[['height_cm', 'weight_kg', 'age']].values

#y -  the target (what we want to predict), 1d Array
y = dataFrame['target_value'].values


# Blank sequential model
# A sequential model is a linear stack of layers where each layer has exactly one input tensor and one output tensor.
model = keras.Sequential()

# Add a layer to the model
# Dense Layer - each neuron is connected to every neuron in the previous layer 
model.add(keras.layers.Dense(
    # number of neurons in the layer
    units = 8, 
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

model.add(
    keras.layers.Dense(
        #one neuron in the layer
        #        / o \
        #        / o \
        #        / o \
        #Start o - o - o End
        #        \ o /
        #        \ o /
        #        \ o /
        #        \ o /
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

history = model.fit(
    # X - the dataset
    X,
    # y - the target (what we want to predict)
    y,
    # number of epochs - how many times to loop over the whole dataset
    epochs=50,
    # batch size - number of examples per gradient update
    batch_size=32,
    # validation split - set aside 20% of data for validation (unseen)
    validation_split=0.2
)


# Graph
import matplotlib.pyplot as plt
plt.plot(history.history['loss'],label = 'train loss')
plt.plot(history.history['val_loss'],label = 'val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()