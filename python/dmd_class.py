import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
import warnings
from numbers import Number

class DMD(object):
    def __init__(self, r=None):
        self.r           = r
        self.A_tilde     = None
        self.phi         = None
        self.lambda_     = None
        self.b           = None
        self.omega       = None
        self.dt          = None
        self.W           = None
        self.n_points    = None
        self.n_snapshots = None
        self.X0          = None

    def fit(self, X, svd_rank=0,dt=1.0):

        self.dt = dt
        self.n_points,self.n_snapshots = X.shape

        X1 = X[:,:-1] 
        X2 = X[:, 1:]
        self.X0 = X1[:,0]

        # Compute SVD of x (uu,ss,vv)
        u,s,v = np.linalg.svd(X1, full_matrices=False)

        # Compute r
        self.r = self.compute_rank(s, X1.shape[0], X1.shape[1], svd_rank)

        # Compute reduced SVD
        print(f'Using rank {self.r}')

        u_r   = u[: , :self.r].conj()
        s_r   = np.diag(s[:self.r])
        v_r   = v[:self.r:, ].conj().transpose()

        s_inv = np.linalg.inv(s_r)

        # Compute Atilde
        self.A_tilde = u_r.T @ X2 @ v_r @ s_inv
        self.lambda_ , self.W = np.linalg.eig(self.A_tilde)
        self.phi = X2 @ v_r @ s_inv @ self.W
        self.b = np.linalg.pinv(self.phi) @ X1[:,0] # amplitude
        self.omega = np.log(self.lambda_) / self.dt

    #TODO: verificar a implementação do predict no pydmd
    #     e comparar com a implementação do dmd_class.py
    def predict(self, tvalues):
        time_dynamics = np.zeros((self.r, len(tvalues)), dtype=complex)
        for i, t in enumerate(tvalues):
            time_dynamics[:, i] = self.b * np.exp(self.omega * t)
        xDMD = self.phi @ time_dynamics
        return xDMD.real
    

    def svht(self,sigma_svd: np.ndarray, rows: int, cols: int) -> int:
        """
        Singular Value Hard Threshold.

        :param sigma_svd: Singual values computed by SVD
        :type sigma_svd: np.ndarray
        :param rows: Number of rows of original data matrix.
        :type rows: int
        :param cols: Number of columns of original data matrix.
        :type cols: int
        :return: Computed rank.
        :rtype: int

        References:
        Gavish, Matan, and David L. Donoho, The optimal hard threshold for
        singular values is, IEEE Transactions on Information Theory 60.8
        (2014): 5040-5053.
        https://ieeexplore.ieee.org/document/6846297
        """
        beta  = np.divide(*sorted((rows, cols)))
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        tau  = np.median(sigma_svd) * omega
        rank = np.sum(sigma_svd > tau)

        if rank == 0:
            warnings.warn(
                "SVD optimal rank is 0. The largest singular values are "
                "indistinguishable from noise. Setting rank truncation to 1.",
                RuntimeWarning,
            )
            rank = 1

        return rank

    def compute_rank(self, s: np.ndarray, rows: Number, cols: Number, svd_rank: Number = 0) -> int:
        """
        References:
        Gavish, Matan, and David L. Donoho, The optimal hard threshold for
        singular values is, IEEE Transactions on Information Theory 60.8
        (2014): 5040-5053.
        """
        if svd_rank == 0:
            rank = self.svht(s, rows, cols)
        elif 0 < svd_rank < 1:
            cumulative_energy = np.cumsum(s**2 / (s**2).sum())
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif svd_rank >= 1 and isinstance(svd_rank, int):
            rank = min(svd_rank, s.size)
        else:
            rank = min(rows, cols)

        return rank
        

