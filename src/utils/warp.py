import numpy as np

class Warp:
    """object that can be called to perform linear affine warps"""
    def __init__(self, p):
        self.p = p
        self.mat = Warp.to_mat(p)

    @staticmethod
    def to_mat(p):
        """
        Convert parameters into (3x3) matrix as follows:

                                      a1 a2 a5
        [a1, a2, a3, a4, a5, a6] <->  a3 a4 a6
                                      0  0  1
        :param p: row of shape (6,)
        :return: matrix of shape (3x3)
        """
        array = np.eye(3, dtype="float64")
        array[:2,:2] = p[:4].reshape((2,2))
        array[:2,2] = p[4:]

        return array

    def __call__(self, x, invert=False, K=None):
        """
        Apply warping x' = pi(W pi^-1(x)) on x. If K is given, applies x' = pi(KWK^-1 pi^-1(x)).
        If invert is given, inverts the matrix.
        :param x: array of points (N,2)
        :param invert: (bool) controlling inversion of matrix
        :param K: intrinsic camera calibration matrix
        :return: array of warped points (N,2)
        """
        # find correct warp
        warp = self.mat if K is None else K.dot(self.mat).dot(np.linalg.inv(K))
        warp = warp if not invert else np.linalg.inv(warp)
        warp = warp.transpose()

        if len(x.shape) == 1:
            # append 1
            x_hom = np.hstack([x, 1])
            x_p_hom = x_hom.dot(warp)

            # reproject
            return x_p_hom[:2] / x_p_hom[2]
        else:
            shape = (x.shape[0],1)
            # append 1
            x_hom = np.hstack([x, np.ones(shape)])
            x_p_hom = x_hom.dot(warp)

            # reproject
            return x_p_hom[:,:2] / x_p_hom[:,2].reshape(shape)