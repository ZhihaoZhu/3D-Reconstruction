import numpy as np
import numpy as np
import helper
import submission
import matplotlib.pyplot as plt
import scipy.optimize
import scipy
from mpl_toolkits.mplot3d import Axes3D



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
x1 = templeCoords["x1"].reshape(-1)
y1 = templeCoords["y1"].reshape(-1)
x2 = x1.copy()
y2 = y1.copy()
I1 = plt.imread('../data/im1.png')
I2 = plt.imread('../data/im2.png')

M = np.max(I1.shape)
F = submission.eightpoint(pts1, pts2, M)


for i in range(len(x1)):


    x2[i], y2[i] = submission.epipolarCorrespondence(I1, I2, F, x1[i], y1[i])


pointset1 = np.concatenate((x1.reshape((-1,1)),y1.reshape((-1,1))),axis=1)
pointset2 = np.concatenate((x2.reshape((-1,1)),y2.reshape((-1,1))),axis=1)



K1 = K['K1']
K2 = K['K2']
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
    w, error = submission.triangulate(C1, pointset1, C2, pointset2)
    if error<error_now:
        error_now = error
        index = i
        C2_save = C2
        w_save = w
    print(error)
print(w_save)
print(w_save.shape)

P = w_save

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xmin, xmax = np.min(P[:, 0]), np.max(P[:, 0])
ymin, ymax = np.min(P[:, 1]), np.max(P[:, 1])
zmin, zmax = np.min(P[:, 2]), np.max(P[:, 2])

ax.set_xlim3d(xmin, xmax)
ax.set_ylim3d(ymin, ymax)
ax.set_zlim3d(zmin, zmax)

ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o')
plt.show()

