import numpy as np
import numpy as np
import helper
import submission
import matplotlib.pyplot as plt
import scipy.optimize
import scipy
from mpl_toolkits.mplot3d import Axes3D

'''
Please un-comment the block to run corresponding question
'''

'''
    Q2_1
'''
# print("Running Question 2_1")
#
# corresp = np.load("../data/some_corresp.npz")
# pts1 = corresp["pts1"]
# pts2 = corresp["pts2"]
# I1 = plt.imread('../data/im1.png')
# I2 = plt.imread('../data/im2.png')
#
# M = np.maximum(I1.shape[0],I1.shape[1])
#
# F = submission.eightpoint(pts1, pts2, M)
# print(F)
# helper.displayEpipolarF(I1, I2, F)

'''
    Q2_2
'''
# print("Running Question 2_2")

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
# F = submission.sevenpoint(pts1, pts2, M)[0]
# np.savez("../results/q2_2.npz", F=F, M=M, pts1=pts1, pts2=pts2)
# print(F)
# helper.displayEpipolarF(I1, I2, F)

'''
    Q3_1
'''
# print("Running Question 3_1")

# F = np.load("../results/q2_1.npz")["F"]
# K = np.load("../data/intrinsics.npz")
# K1 = K['K1']
# K2 = K['K2']
# E = submission.essentialMatrix(F, K1, K2)
# print(E)

'''
    Q4_1
'''

# print("Running Question 4_1")

#
# I1 = plt.imread('../data/im1.png')
# I2 = plt.imread('../data/im2.png')
#
# K = np.load("../data/intrinsics.npz")
# corresp = np.load("../data/some_corresp.npz")
# pts1 = corresp["pts1"]
# pts2 = corresp["pts2"]
# M = np.max(I1.shape)
# K1 = K['K1']
# K2 = K['K2']
# F = submission.eightpoint(pts1, pts2, M)
# helper.epipolarMatchGUI(I1, I2, F)

'''
    Q5_1
'''

# print("Running Question 5_1")

# corresp = np.load("../data/some_corresp_noisy.npz")
# pts1 = corresp["pts1"]
# pts2 = corresp["pts2"]
#
# I1 = plt.imread('../data/im1.png')
# I2 = plt.imread('../data/im2.png')
# M = np.maximum(I1.shape[0],I1.shape[1])
#
# F_ransac = submission.ransacF(pts1, pts2, M)
# F_eight = submission.eightpoint(pts1, pts2, M)
#
# '''
#     Replace with F_eight
# '''
# helper.displayEpipolarF(I1, I2, F_ransac)



'''
    Q5_3
'''
print("Running Question 5_3")
corresp = np.load("../data/some_corresp_noisy.npz")
pts1 = corresp["pts1"]
pts2 = corresp["pts2"]
I1 = plt.imread('../data/im1.png')
I2 = plt.imread('../data/im2.png')
M = np.maximum(I1.shape[0],I1.shape[1])
F, pts1_in, pts2_in = submission.ransacF(pts1, pts2, M)
# np.savez("../data/F_ransac.npz", F=F, pts1_in=pts1_in, pts2_in=pts2_in)
# F = np.load("../data/F_ransac.npz")["F"]
# pts1_in = np.load("../data/F_ransac.npz")["pts1_in"]
# pts2_in = np.load("../data/F_ransac.npz")["pts2_in"]

K = np.load("../data/intrinsics.npz")
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
    w, error = submission.triangulate(C1, pts1_in, C2, pts2_in)
    if error<error_now:
        if np.min(w[:,2])>0:
            error_now = error
            index = i
            C2_save = C2
            w_save = w
print('Reprojection error before Bundle Adjustment: %f' % error_now)
M2 = M2s[:, :, index]

M2_final, w_final = submission.bundleAdjustment(K1, M1, pts1_in, K2, M2, pts1_in, w_save[:,0:3])

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


















