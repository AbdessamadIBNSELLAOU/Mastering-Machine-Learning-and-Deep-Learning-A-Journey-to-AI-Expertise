import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils2 import *

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([
        # ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
        tf.keras.layers.ZeroPadding2D(padding=(3, 3), input_shape=(64, 64, 3)),
        
        # Conv2D with 32 7x7 filters and stride of 1
        tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1)),
        
        # BatchNormalization on axis 3
        tf.keras.layers.BatchNormalization(axis=3),
        
        # ReLU activation
        tf.keras.layers.ReLU(),
        
        # MaxPooling2D with default pool size (2, 2)
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten layer
        tf.keras.layers.Flatten(),
        
        # Dense layer with 1 unit and sigmoid activation
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    
    return model

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Arguments:
    input_shape -- shape of the input images, e.g. (64, 64, 3)

    Returns:
    model -- Keras model
    """

    input_img = tf.keras.Input(shape=input_shape)
    
    # CONV2D: 8 filters 4x4, stride 1, padding SAME
    Z1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=1, padding='same')(input_img)
    
    # RELU
    A1 = tf.keras.layers.ReLU()(Z1)
    
    # MAXPOOL: window 8x8, stride 8, padding SAME
    P1 = tf.keras.layers.MaxPooling2D(pool_size=(8, 8), strides=8, padding='same')(A1)
    
    # CONV2D: 16 filters 2x2, stride 1, padding SAME
    Z2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=1, padding='same')(P1)
    
    # RELU
    A2 = tf.keras.layers.ReLU()(Z2)
    
    # MAXPOOL: window 4x4, stride 4, padding SAME
    P2 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(A2)
    
    # FLATTEN
    F = tf.keras.layers.Flatten()(P2)
    
    # Dense layer with 6 units and softmax activation
    outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)
    
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model