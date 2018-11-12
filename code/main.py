import numpy as np
import numpy as np
import helper
import submission
import matplotlib.pyplot as plt
import scipy.optimize
import scipy
from mpl_toolkits.mplot3d import Axes3D


'''
    Q2_1
'''
# corresp = np.load("../data/some_corresp.npz")
# pts1 = corresp["pts1"]
# pts2 = corresp["pts2"]
# I1 = plt.imread('../data/im1.png')
# I2 = plt.imread('../data/im2.png')
#
# M = np.maximum(I1.shape[0],I1.shape[1])
#
# F = submission.eightpoint(pts1, pts2, M)
# helper.displayEpipolarF(I1, I2, F)

'''
    Q2_2
'''
# corresp = np.load("../data/some_corresp.npz")
# pts1 = corresp["pts1"]
# pts2 = corresp["pts2"]
#
# select_ids = [53, 17, 43, 46, 27, 56, 78]
# pts1 = pts1[select_ids, :]
# pts2 = pts2[select_ids, :]
#
# I1 = plt.imread('../data/im1.png')
# I2 = plt.imread('../data/im2.png')
#
# M = np.maximum(I1.shape[0],I1.shape[1])
#
# F = submission.sevenpoint(pts1, pts2, M)[0]
#
# np.savez("../results/q2_2.npz", F=F, M=M, pts1=pts1, pts2=pts2)
#
# helper.displayEpipolarF(I1, I2, F)

'''
    Q3_1
'''
# F = np.load("../results/q2_1.npz")["F"]
# K = np.load("../data/intrinsics.npz")
# K1 = K['K1']
# K2 = K['K2']
# E = submission.essentialMatrix(F, K1, K2)

'''
    Q3_2 & Q3_3
'''
# F = np.load("../results/q2_1.npz")["F"]
# K = np.load("../data/intrinsics.npz")
# corresp = np.load("../data/some_corresp.npz")
# pts1 = corresp["pts1"]
# pts2 = corresp["pts2"]
# K1 = K['K1']
# K2 = K['K2']
# E = submission.essentialMatrix(F, K1, K2)
#
# M2s = helper.camera2(E)
# M1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
# C1 = K1@M1
#
# index = 0
# error_now = 10000000
# for i in range(4):
#     C2 = K2 @ M2s[:, :, i]  # To be modified
#     w, error = submission.triangulate(C1, pts1, C2, pts2)
#     print(error)
#     if error<error_now:
#         index = i
#         C2_save = C2
#         w_save = w

# np.savez("../results/q3_3.npz", M2=M2s[:, :, index], C2=C2_save, P=w_save)


'''
    Q4_1
'''
#
# I1 = plt.imread('../data/im1.png')
# I2 = plt.imread('../data/im2.png')
# # I1 = np.dot(I1[...,:3], [0.299, 0.587, 0.114])
# # I2 = np.dot(I2[...,:3], [0.299, 0.587, 0.114])
#
# K = np.load("../data/intrinsics.npz")
# corresp = np.load("../data/some_corresp.npz")
# pts1 = corresp["pts1"]
# pts2 = corresp["pts2"]
# M = np.max(I1.shape)
# K1 = K['K1']
# K2 = K['K2']
# F = submission.eightpoint(pts1, pts2, M)
#
# helper.epipolarMatchGUI(I1, I2, F)

'''
    Q5_1
'''


# corresp = np.load("../data/some_corresp_noisy.npz")
# pts1 = corresp["pts1"]
# pts2 = corresp["pts2"]
#
# I1 = plt.imread('../data/im1.png')
# I2 = plt.imread('../data/im2.png')
# M = np.maximum(I1.shape[0],I1.shape[1])
#
# F = submission.ransacF(pts1, pts2, M)
#
# # np.savez("../results/q5_1.npz", F=F, M=M, pts1=pts1, pts2=pts2)
#
# helper.displayEpipolarF(I1, I2, F)



'''
    Q5_3
'''

corresp = np.load("../data/some_corresp_noisy.npz")
pts1 = corresp["pts1"]
pts2 = corresp["pts2"]
K = np.load("../data/intrinsics.npz")
K1 = K['K1']
K2 = K['K2']

I1 = plt.imread('../data/im1.png')
I2 = plt.imread('../data/im2.png')
M = np.maximum(I1.shape[0],I1.shape[1])

'''
    Get F
'''
F, pts1_in, pts2_in = submission.ransacF(pts1, pts2, M)

# just to test
corresp = np.load("../data/some_corresp.npz")
pts1 = corresp["pts1"]
pts2 = corresp["pts2"]
F = submission.eightpoint(pts1, pts2, M)


'''
    Get M2
'''
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
    w, error = submission.triangulate(C1, pts1_in, C2, pts2_in)
    if error<error_now:
        error_now = error
        index = i
        C2_save = C2
        w_save = w
M2=M2s[:, :, index]
C2 = K2 @ M2

print("yes")

'''
    Get temple data
'''

templeCoords = np.load("../data/templeCoords.npz")
x1 = templeCoords["x1"].reshape(-1)
y1 = templeCoords["y1"].reshape(-1)
x2 = x1.copy()
y2 = y1.copy()
I1 = plt.imread('../data/im1.png')
I2 = plt.imread('../data/im2.png')
for i in range(len(x1)):
    x2[i], y2[i] = submission.epipolarCorrespondence(I1, I2, F, x1[i], y1[i])
pointset1 = np.concatenate((x1.reshape((-1,1)),y1.reshape((-1,1))),axis=1)
pointset2 = np.concatenate((x2.reshape((-1,1)),y2.reshape((-1,1))),axis=1)
w, error = submission.triangulate(C1, pointset1, C2, pointset2)
M2_final, w_final = submission.bundleAdjustment(K1, M1, pointset1, K2, M2, pointset2, w[:,0:3])
print(M2_final)

P = w_final
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















