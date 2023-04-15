import numpy as np

def softmax(x: np.ndarray, axis = 0):
    e_x = np.exp(x - np.max(x, axis = axis, keepdims = True))    
    sm  = np.sum(e_x, axis = axis, keepdims = True)    
    return e_x / sm    