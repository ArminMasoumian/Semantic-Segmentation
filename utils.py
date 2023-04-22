import numpy as np

def load_data():
    # Load the training and validation data
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_val = np.load('x_val.npy')
    y_val = np.load('y_val.npy')
    
    # Preprocess the data
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    
    return x_train, y_train, x_val, y_val
