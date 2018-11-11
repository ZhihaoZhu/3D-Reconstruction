import numpy as np
import numpy as np
import helper
import submission
import matplotlib.pyplot as plt
import scipy.optimize
import scipy
import sympy as sp

I1 = plt.imread('../data/im1.png')
I2 = plt.imread('../data/im2.png')

K = np.load("../data/intrinsics.npz")
corresp = np.load("../data/some_corresp.npz")
pts1 = corresp["pts1"]
pts2 = corresp["pts2"]
M = np.max(I1.shape)
K1 = K['K1']
K2 = K['K2']
F = submission.eightpoint(pts1, pts2, M)
xc = 150
yc = 200

x2, y2 = submission.epipolarCorrespondence(I1, I2, F, xc, yc)

print(x2,y2)



