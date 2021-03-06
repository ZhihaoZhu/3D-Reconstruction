import numpy as np
import numpy as np
import helper
import submission
import matplotlib.pyplot as plt
import scipy.optimize
import scipy
from mpl_toolkits.mplot3d import Axes3D


'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, p1, p2, R and P to q3_3.mat
'''

I1 = plt.imread('../data/im1.png')

K = np.load("../data/intrinsics.npz")
corresp = np.load("../data/some_corresp.npz")
pts1 = corresp["pts1"]
pts2 = corresp["pts2"]
M = np.max(I1.shape)

K1 = K['K1']
K2 = K['K2']
F = submission.eightpoint(pts1, pts2, M)


E = submission.essentialMatrix(F, K1, K2)
M2s = helper.camera2(E)
M1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
C1 = K1@M1

index = 0
error_now = 10000000
C2_save = 0
w_save = 0
for i in range(4):
    C2 = K2 @ M2s[:, :, i]  # To be modified
    w, error = submission.triangulate(C1, pts1, C2, pts2)
    if error<error_now:
        error_now = error
        index = i
        C2_save = C2
        w_save = w
    print(error)

print("best error result:", error_now)

np.savez("../results/q3_3.npz", M2=M2s[:, :, index], C2=C2_save, P=w_save)



