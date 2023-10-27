import numpy as np
from numpy import sin

# Code for checkerboard pattern distribution from https://stackoverflow.com/questions/60019462/create-checkerboard-distribution-with-python
def chessboard_distribution(n_points=10000, n=6):
    n_classes = 2
    n = 6
    x = np.random.uniform(-n,n, size=(n_points, n_classes))
    mask = np.logical_or(np.logical_and(sin(x[:,0]) > 0.0, sin(x[:,1]) > 0.0), \
    np.logical_and(sin(x[:,0]) < 0.0, sin(x[:,1]) < 0.0))
    y = np.eye(n_classes)[1*mask]
    return x[y[:, 0]==1]