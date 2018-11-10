import numpy as np
import helper
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

    pts1_o = pts1.copy()
    pts2_o = pts2.copy()

    pts1 = pts1/M
    pts2 = pts2/M
    A = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        # cord_1 = np.array([pts1[i, 1], pts1[i, 0], 1]).reshape((1, 3))
        # row = np.concatenate((pts2[i, 1] * cord_1, pts2[i, 0] * cord_1), axis=1)
        # row = np.concatenate((row, 1 * cord_1), axis=1)

        cord_1 = np.array([pts1[i, 0], pts1[i, 1], 1]).reshape((1, 3))
        row = np.concatenate((pts2[i, 0] * cord_1, pts2[i, 1] * cord_1), axis=1)
        row = np.concatenate((row, 1 * cord_1), axis=1)

        A[i, :] = row.reshape(-1)
    u, s, vh = np.linalg.svd(A)
    F = vh.transpose()[:,-1].reshape((3,3))
    # F = np.transpose(vh)[:,-1].reshape((3,3))
    '''
        the singularity constraint
    '''
    u, s, vh = np.linalg.svd(F)
    ss = np.eye(3)
    ss[0,0] = s[0]
    ss[1,1] = s[1]
    ss[2,2] = 0
    F = u.dot(ss).dot(vh)

    # F = helper.refineF(F, pts1, pts2)
    F = helper.refineF(F, pts1_o, pts2_o)
    F = F/M/M
    # F = F*M*M

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
    # Replace pass by your implementation
    pass


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    pass


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
    # Replace pass by your implementation
    pass


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
    # Replace pass by your implementation
    pass

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    pass

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
