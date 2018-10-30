import numpy as np
import numpy as np
import helper
import submission
import matplotlib.pyplot as plt
import scipy.optimize

corresp = np.load("../data/some_corresp.npz")
pts1 = corresp["pts1"]
pts2 = corresp["pts2"]

I1 = plt.imread('../data/im1.png')
I2 = plt.imread('../data/im2.png')
print(I1.shape)
print(I2.shape)
M = np.maximum(I1.shape[0],I1.shape[1])
print(M)

F = submission.eightpoint(pts1, pts2, M)
helper.displayEpipolarF(I1, I2, F)