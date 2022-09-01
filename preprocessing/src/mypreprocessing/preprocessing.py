import numpy as np


def preprocess(data):
    
    # Convert data to the format you need, the following is a gimmick 
    # here, you can apply scaler, do one-hot encoding and other things
    x = np.array(data)

    return x