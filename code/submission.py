import numpy as np
import helper
import sympy as sp
import scipy
import scipy.stats as st


"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    '''
        Need to check the alternatives
    '''

    pts1 = pts1/M
    pts2 = pts2/M


    A = np.zeros((pts1.shape[0], 9))
    x = 0
    y = 1
    for i in range(pts1.shape[0]):
        A[i, 0:3] = np.array([pts1[i, x], pts1[i, y], 1]) * pts2[i, x]
        A[i, 3:6] = np.array([pts1[i, x], pts1[i, y], 1]) * pts2[i, y]
        A[i, 6:9] = np.array([pts1[i, x], pts1[i, y], 1])

    # print("A",A)
    u, s, vh = np.linalg.svd(A)

    F = vh.transpose()[:,-1].reshape((3,3))

    u, s, vh = np.linalg.svd(F)

    ss = np.eye(3)
    ss[0,0] = s[0]
    ss[1,1] = s[1]
    ss[2,2] = 0
    F = u.dot(ss).dot(vh)
    # F = helper.refineF(F, pts1, pts2)
    T = np.zeros((3, 3), dtype=np.float32)
    T[0, 0] = 1.0 / M
    T[1, 1] = 1.0 / M
    T[2, 2] = 1.0
    F = T.transpose() @ F @ T
    np.savez("../results/q2_1.npz", F = F, M = M)

    return F

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    pts1 = pts1 / M
    pts2 = pts2 / M

    A = np.zeros((pts1.shape[0], 9))
    x = 0
    y = 1
    for i in range(pts1.shape[0]):
        A[i, 0:3] = np.array([pts1[i, x], pts1[i, y], 1]) * pts2[i, x]
        A[i, 3:6] = np.array([pts1[i, x], pts1[i, y], 1]) * pts2[i, y]
        A[i, 6:9] = np.array([pts1[i, x], pts1[i, y], 1])

    u, s, vh = np.linalg.svd(A)

    f1 = vh.transpose()[:, -1].reshape((3, 3))
    f2 = vh.transpose()[:, -2].reshape((3, 3))

    a = sp.Symbol('a')

    FF = f1*a + f2*(1-a)
    F = sp.Matrix(FF)
    det = F.det()
    alpha = sp.solve(det, a)

    Farray = []
    for i in range(len(alpha)):
        F = f1*float(alpha[i].as_real_imag()[0]) + f2*(1-float(alpha[i].as_real_imag()[0]))
        u, s, vh = np.linalg.svd(F)

        ss = np.eye(3)
        ss[0, 0] = s[0]
        ss[1, 1] = s[1]
        ss[2, 2] = s[2]
        F = u.dot(ss).dot(vh)
        # F = helper.refineF(F, pts1, pts2)
        T = np.zeros((3, 3), dtype=np.float32)
        T[0, 0] = 1.0 / M
        T[1, 1] = 1.0 / M
        T[2, 2] = 1.0
        F = T.transpose() @ F @ T
        Farray.append(F)


    return Farray


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = K2.transpose() @ F @ K1
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    A = np.zeros((pts1.shape[0],4,4))
    w = np.zeros((pts1.shape[0],4))
    error = 0

    for i in range(pts1.shape[0]):

        A[i, 0, :] = C1[0, :] - pts1[i, 0] * C1[2, :]
        A[i, 1, :] = C1[1, :] - pts1[i, 1] * C1[2, :]
        A[i, 2, :] = C2[0, :] - pts2[i, 0] * C2[2, :]
        A[i, 3, :] = C2[1, :] - pts2[i, 1] * C2[2, :]

        u, s, vh = np.linalg.svd(A[i])
        # print(A[i])
        w_l = vh.transpose()[:, -1]
        w_l[0:3] = w_l[0:3]/w_l[3]
        w_l[3] = 1
        w[i,:] = w_l


    for i in range(pts1.shape[0]):
        point1 = C1@w[i]
        x = np.array([point1[0]/point1[2], point1[1]/point1[2]])


        l1 = np.linalg.norm(x-pts1[i])**2
        point2 = C2 @ w[i]
        x = np.array([point2[0] / point2[2], point2[1] / point2[2]])
        l2 = np.linalg.norm(x-pts2[i])**2
        error = error + l1 + l2

    return w, error


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''


def epipolarCorrespondence(im1, im2, F, x1, y1):
    X1 = np.array([x1,y1,1])
    window_size = 1
    cord = F@X1
    width = im1.shape[1]
    height = im1.shape[0]
    y = np.arange(height).astype(int)
    x = np.arange(height).astype(int)
    for i in range(height):
        x[i] = np.round((-cord[2]-y[i]*cord[1])/cord[0])
        if x[i]<0 or x[i]>width:
            x[i] = 666

    kernlen = window_size*2+1
    nsig = 3
    interval = (2 * nsig + 1.) / (kernlen)
    p = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(p))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()

    template = im1[y1-window_size:y1+window_size+1, x1-window_size:x1+window_size+1]
    err = 10000
    x2_o = 0
    y2_o = 0
    for i in range(3,height-3):
        x2 = x[i]
        y2 = y[i]
        window = im2[y2-window_size:y2+window_size+1, x2-window_size:x2+window_size+1]

        dist = np.linalg.norm((template-window)*kernel)
        if dist<err:
            x2_o = x2
            y2_o = y2
            err = dist

    return x2_o, y2_o



'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
'''
def ransacF(pts1, pts2, M):
    iter_num = 50
    index_max = 0
    threshold = 0.001
    F_final = np.zeros((3,3))
    p1_final = []
    p2_final = []
    for i in range(iter_num):
        print(i)
        ids = np.random.randint(low=0, high=pts1.shape[0], size=7)
        pset1 = pts1[ids, :]
        pset2 = pts2[ids, :]
        Farray = sevenpoint(pset1, pset2, M)
        for j in range(len(Farray)):
            F = Farray[j]
            index = 0
            p1_to_store = []
            p2_to_store = []

            for g in range(pts1.shape[0]):
                p1 = np.array([pts1[g,0], pts1[g,1], 1])
                p2 = np.array([pts2[g,0], pts2[g,1], 1])
                err = abs(p2.transpose() @ F @ p1)
                if err<threshold:
                    index = index+1
                    p1_to_store.append([pts1[g,0], pts1[g,1]])
                    p2_to_store.append([pts2[g,0], pts2[g,1]])

            # print("index", index)
            if index > index_max:
                p1_final = np.array(p1_to_store)
                p2_final = np.array(p2_to_store)
                index_max = index
                print(index_max)
                F_final = F

    F_output = eightpoint(p1_final, p2_final, M)


    return F_output







'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
