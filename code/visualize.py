import numpy as np
import numpy as np
import helper
import submission
import matplotlib.pyplot as plt
import scipy.optimize
import scipy

'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter3
'''

templeCoords = np.load("../data/templeCoords.npz")
corresp = np.load("../data/some_corresp.npz")
K = np.load("../data/intrinsics.npz")

pts1 = corresp["pts1"]
pts2 = corresp["pts2"]
x1 = templeCoords["x1"]
y1 = templeCoords["y1"]
x2 = x1
y2 = y1
I1 = plt.imread('../data/im1.png')
I2 = plt.imread('../data/im2.png')

M = np.max(I1.shape)
F = submission.eightpoint(pts1, pts2, M)

for i in range(x1.shape):
    x2[i], y2[i] = submission.epipolarCorrespondence(I1, I2, F, x1, y1)

print(x2)
#
# K1 = K['K1']
# K2 = K['K2']
# E = submission.essentialMatrix(F, K1, K2)
# M2s = helper.camera2(E)
# M1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1]])
# C1 = K1@M1
#
# index = 0
# error_now = 10000
# for i in range(4):
#     C2 = K2 @ M2s[:, :, i]  # To be modified
#     w, error = submission.triangulate(C1, pts1, C2, pts2)
#     print(error)
#     if error<error_now:
#         index = i
#         C2_save = C2
#         w_save = w
#
# print(w_save)




