import numpy as np
import numpy as np
import helper
import submission
import matplotlib.pyplot as plt
import scipy.optimize
import scipy

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
F = np.load("../results/q2_1.npz")["F"]
K = np.load("../data/intrinsics.npz")
K1 = K['K1']
K2 = K['K2']
E = submission.essentialMatrix(F, K1, K2)









