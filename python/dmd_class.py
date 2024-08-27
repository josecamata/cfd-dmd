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
        self.eigvectors  = None
        self.n_points    = None
        self.n_snapshots = None
        self.X0          = None
        self.parameters = dict()
        self.parameters["svd_type"]              = "svd"
        self.parameters["rsvd_base_vector_size"] = 50
        self.parameters["rsvd_oversampling"]     = 20
        self.parameters["rsvd_power_iters"]      = 2
        self.parameters["rsvd_seed"]             = 1
        self.parameters["dt"]            = 1
        self.parameters["t0"]            = 0
        self.parameters["tend"]          = 0
  
    

    def compute_svd(self, mat, svd_rank):

        self.r  = self.compute_rank(mat,svd_rank)

        if(self.parameters["svd_type"] == "svd"):
            print("Using SVD factorization\n")
            [U,s,V] = np.linalg.svd(mat,full_matrices=False, compute_uv=True, hermitian=False)
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
            [U_tilde, s, V] = np.linalg.svd(B,full_matrices=False)
            U = Q @ U_tilde

        V = V.conj().T

        U = U[:, :self.r]
        V = V[:, :self.r]
        s = s[:self.r]

        return U, s, V

    def fit(self, X, svd_rank=0):

        self.n_points,self.n_snapshots = X.shape

        self.parameters["tend"] = self.n_snapshots
        dt = self.parameters["dt"]

        X1 = X[:,:-1] 
        X2 = X[:, 1:]
        self.X0 = X1[:,0]

        # Compute SVD of x (uu,ss,vv)
        U,s,V = self.compute_svd(X1, svd_rank)
        

        # Compute Atilde
        #self.A_tilde = np.linalg.multi_dot([u_r,X2,v_r,s_r]);
        self.A_tilde = np.linalg.multi_dot([U.T.conj(), X2, V]) * np.reciprocal(s)

        self.eigvalues , self.eigvectors = np.linalg.eig(self.A_tilde)

        # self.phi = X2 @ v_r @ s_r @ self.W
        self.phi = (X2.dot(V) * np.reciprocal(s)).dot(self.eigvectors)
        #self.phi = np.linalg.multi_dot([X2 , V , s_r , self.eigvectors])

        self.b = np.linalg.pinv(self.phi) @ self.X0 # amplitude
        #self.b = np.linalg.lstsq(self.phi,self.X0.T, rcond=None )[0]
        self.b = self.b[:,np.newaxis]
        self.omega = np.log(self.eigvalues) / ( dt )
        self.omega = self.omega[:, np.newaxis]


    def predict(self, tvalues):
        tvalues = tvalues[np.newaxis,:]
        temp = np.multiply(self.omega, tvalues)
        temp  = np.exp(temp)
        time_dynamics = np.multiply(self.b, temp)
        xDMD = np.dot(self.phi,time_dynamics)
        return xDMD.real
    
    def reconstructed_data(self):
        t_values = self.dmd_timesteps()
        t_values = t_values[np.newaxis,:]
        temp     = np.multiply(self.omega, t_values)
        temp     = np.exp(temp)
        time_dynamics = np.multiply(self.b, temp)
        xDMD          = np.dot(self.phi,time_dynamics)
        return xDMD.real

    def dmd_timesteps(self):
        """
        Get the timesteps of the reconstructed states.

        :return: the time intervals of the original snapshots.
        :rtype: numpy.ndarray
        """
        return np.arange(
            self.parameters["t0"],
            self.parameters["tend"],
            self.parameters["dt"],
        )

    def compute_rank(self, X, svd_rank=0):
        """
        Rank computation for the truncated Singular Value Decomposition.
        :param numpy.ndarray X: the matrix to decompose.
        :param svd_rank: the rank for the truncation; If 0, the method computes
            the optimal rank and uses it for truncation; if positive interger,
            the method uses the argument for the truncation; if float between 0
            and 1, the rank is the number of the biggest singular values that
            are needed to reach the 'energy' specified by `svd_rank`; if -1,
            the method does not compute truncation. Default is 0.
        :type svd_rank: int or float
        :return: the computed rank truncation.
        :rtype: int
        References:
        Gavish, Matan, and David L. Donoho, The optimal hard threshold for
        singular values is, IEEE Transactions on Information Theory 60.8
        (2014): 5040-5053.
        """
        U, s, _ = np.linalg.svd(X, full_matrices=False)

        def omega(x):
            return 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43

        if svd_rank == 0:
            beta = np.divide(*sorted(X.shape))
            tau = np.median(s) * omega(beta)
            rank = np.sum(s > tau)
            if rank == 0:
                warnings.warn(
                    "SVD optimal rank is 0. The largest singular values are "
                    "indistinguishable from noise. Setting rank truncation to 1.",
                    RuntimeWarning,
                )
                rank = 1
        elif 0 < svd_rank < 1:
            cumulative_energy = np.cumsum(s**2 / (s**2).sum())
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif svd_rank >= 1 and isinstance(svd_rank, int):
            rank = min(svd_rank, U.shape[1])
        else:
            rank = min(X.shape)

        return rank

