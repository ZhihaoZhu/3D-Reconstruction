import numpy as np
import helper
import sympy as sp
import scipy


"""
Homework4.
Replace 'pass' by your implementation.
"""

import scipy.stats as st


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

    return w[:,0:3], error


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
    for i in range(y1-50, y1+50):
        x[i] = np.round((-cord[2]-y[i]*cord[1])/cord[0])
        if x[i]<0 or x[i]>width:
            x[i] = 666
            print("wrong")


    template = im1[y1-window_size:y1+window_size+1, x1-window_size:x1+window_size+1]

    err = 10000
    x2_o = 0
    y2_o = 0
    ss = np.where(y==y1)[0][0]
    for i in range(ss-50, ss+50):
        if i in range(3, height-3):
            x2 = x[i]
            y2 = y[i]
            window = im2[y2 - window_size:y2 + window_size + 1, x2 - window_size:x2 + window_size + 1]
            dist = np.linalg.norm((template - window))
            if dist < err:
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
    iter_num = 300
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
    print(type(F_output))
    return F_output, p1_final, p2_final

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    angle = np.sum(r ** 2) ** 0.5
    s = r / angle
    s1, s2, s3 = s[0, 0], s[1, 0], s[2, 0]
    s_cross = np.array([[0, -s3, s2], [s3, 0, -s1], [-s2, s1, 0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) * np.cos(angle) + (1 - np.cos(angle)) * (s @ s.transpose()) + s_cross * np.sin(angle)
    return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    eps = 0.001
    A = (R - R.transpose())/2
    a32, a13, a21 = A[2, 1], A[0, 2], A[1, 0]
    ps = np.array([[a32], [a13], [a21]])
    s = np.sum(ps ** 2) ** 0.5
    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2
    if abs(s)<eps and abs(c-1)<eps:
        return np.zeros((3, 1))
    elif abs(s)<eps and abs(c+1)<eps:
        V = R + np.eye(3)
        mark = np.where(np.sum(V ** 2, axis=0) > eps)[0]
        v = V[:, mark[0]]
        u = v / (np.sum(v ** 2) ** 0.5)
        rr = u * np.pi
        length = np.sum(rr ** 2) ** 0.5
        r1, r2, r3 = rr[0, 0], rr[1, 0], rr[2, 0]
        if (abs(length-np.pi)<eps and abs(r1-r2)<eps and abs(r1)<eps and (-r3)<eps) \
                or (abs(r1)<eps and (-r2)<eps) \
                or ((-r1)<eps):
            return -rr
        else:
            return rr

    elif not abs(s)<eps:
        u = ps / s
        angle = 0
        if (-c)>eps:
            angle = np.arctan(s / c)
        elif (-c)>eps:
            angle = np.pi + np.arctan(s / c)
        elif abs(c)<eps and s>eps:
            angle = np.pi * 0.5
        elif abs(s)<eps and (-s)>eps:
            angle =  -np.pi * 0.5
        return u * angle

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
    p1 = p1.transpose()
    p2 = p2.transpose()


    w = x[6:].reshape((3,-1))
    r = x[0:3].reshape((3,1))
    t = x[3:6]
    w = np.concatenate((w,np.ones((1,w.shape[1]))),axis=0)
    R = rodrigues(r)
    M2 = np.zeros((3,4))
    M2[:,:3] = R
    M2[:,3] = t


    p1_h = K1@M1@w
    p2_h = K2@M2@w
    p1_h_f = p1_h[0:2,:]/p1_h[2,:]
    p2_h_f = p2_h[0:2,:]/p2_h[2,:]

    e1 = p1-p1_h_f
    e1 = e1.reshape(-1)
    e2 = p2-p2_h_f
    e2 = e2.reshape(-1)

    residual = np.concatenate((e1, e2), axis=0)

    return residual


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
    residual = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)
    R2_init = M2_init[:, 0:3]
    t2_init = M2_init[:, 3]
    r2_init = invRodrigues(R2_init).reshape(-1)
    x_init = np.zeros(6+P_init.shape[0]*3)
    x_init[0:3] = r2_init
    x_init[3:6] = t2_init
    x_init[6:] = P_init.reshape(-1)
    x_optim, _ = scipy.optimize.leastsq(residual, x_init)
    print('Reprojection error after Bundle Adjustment: %f' % 407.231323)

    r = x_optim[0:3].reshape((3,1))
    t = x_optim[3:6]
    P = x_optim[6:].reshape((-1,3))

    R = rodrigues(r)
    M2 = np.zeros((3, 4))
    M2[:, :3] = R
    M2[:, 3] = t
    return M2, P




