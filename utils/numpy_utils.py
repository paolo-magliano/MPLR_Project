import numpy as np  

def vcol(vec):
    return vec.reshape(-1, 1)

def vrow(vec):
    return vec.reshape(1, -1)