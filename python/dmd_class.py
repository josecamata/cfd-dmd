import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
import warnings
from numbers import Number

class DMD(object):
    def __init__(self, svd_type=0):
        self.r           = None
        self.A_tilde     = None
        self.phi         = None
        self.eigvalues   = None
        self.b           = None
        self.omega       = None
        self.eigvectors   = None
        self.n_points    = None
        self.n_snapshots = None
        self.X0          = None
        self.parameters = dict()
        self.parameters["svd_type"]              = "svd"
        self.parameters["rsvd_base_vector_size"] = 50
        self.parameters["rsvd_oversampling"]     = 20
        self.parameters["rsvd_power_iters"]      = 2
        self.parameters["rsvd_seed"]             = 1
        self.parameters["time_interval"]         = 1
        self.parameters["start_snapshot"]        = 0
        self.parameters["end_snapshot"]          = 0
  
    

    def compute_svd(self, mat):
        if(self.parameters["svd_type"] == "svd"):
            print("Using SVD factorization\n")
            [u,s,v] = np.linalg.svd(mat,full_matrices=False, compute_uv=True, hermitian=False)
        else:
            print("Using rSVD factorization\n")
            m                = mat.shape[1]
            basis_vectors    = self.parameters["rsvd_base_vector_size"]
            oversampling     = self.parameters["rsvd_oversampling"]
            rsvd_power_iters = self.parameters["rsvd_power_iters"]
            seed             = self.parameters["rsvd_seed"]
            
            O = np.random.randn(m, basis_vectors + oversampling)
            Y = mat.dot(O)
            Q, _ = np.linalg.qr(Y)
            for _ in range(rsvd_power_iters):
                Z, _ = np.linalg.qr(mat.T @ Q)
                Q, _ = np.linalg.qr(mat @ Z)

            B = Q.T @ mat
            [u_tilde, s, v] = np.linalg.svd(B,full_matrices=False)
            u = Q @ u_tilde
        return u,s,v

    def fit(self, X, svd_rank=0,dt=1.0):

        self.dt = dt
        self.n_points,self.n_snapshots = X.shape

        self.parameters["end_snapshot"] = self.n_snapshots

        X1 = X[:,:-1] 
        X2 = X[:, 1:]
        self.X0 = X1[:,0]

        # Compute SVD of x (uu,ss,vv)
        u,s,v = self.compute_svd(X1)
        
        # Compute r
        self.r = self.compute_rank(s, X1.shape[0], X1.shape[1], svd_rank)

        # Compute reduced SVD
        print(f'Using rank {self.r}')

        u_r   = np.transpose(u[: , 0:self.r])
        s_r   = s[0:self.r]
        s_r   = np.divide(1.0, s_r)
        s_r   = np.diag(s_r)
        v_r   = np.transpose(v[0:self.r:, ])

        # Compute Atilde
        self.A_tilde = np.linalg.multi_dot([u_r,X2,v_r,s_r]);

        self.eigvalues , self.eigvectors = np.linalg.eig(self.A_tilde)

        # self.phi = X2 @ v_r @ s_r @ self.W

        self.phi = np.linalg.multi_dot([X2 , v_r , s_r , self.eigvectors])

        self.b = np.linalg.pinv(self.phi) @ self.X0 # amplitude
        self.b = self.b[:,np.newaxis]
        self.omega = np.log(self.eigvalues) / (self.dt )
        self.omega = self.omega[:, np.newaxis]

    #TODO: verificar a implementação do predict no pydmd
    #     e comparar com a implementação do dmd_class.py
    def predict(self, tvalues):
        tvalues = tvalues[np.newaxis,:]
        temp = np.multiply(self.omega, tvalues)
        temp  = np.exp(temp)
        time_dynamics = np.multiply(self.b, temp)
        xDMD = np.dot(self.phi,time_dynamics)
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
        

