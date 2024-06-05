import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

class DMD(object):
    def __init__(self, r=None):
        self.r = r
        self.A_tilde = None
        self.phi    = None
        self.lambda_ = None
        self.b      = None
        self.omega  = None
        self.dt     = None
        self.W      = None
        self.n_points = None
        self.n_snapshots = None
        self.X0 = None

    def fit(self, X, thresh=0.7,dt=1.0, rank=None):

        self.dt = dt
        self.n_points,self.n_snapshots = X.shape

        X1 = X[:,:-1] 
        X2 = X[:, 1:]
        self.X0 = X1[:,0]

        # Compute SVD of x (uu,ss,vv)
        u,s,v = np.linalg.svd(X1, full_matrices=False)

        # Compute r
        if rank is not None:
            self.r = rank
        else:
            q = np.cumsum(s) / np.sum(s)
            mask = q > thresh
            self.r = np.where(mask)[0][0]


        u_r   = u[: , :self.r].conj()
        s_r   = np.diag(s[:self.r])
        v_r   = v[:self.r:, ].conj().transpose()

        s_inv = np.linalg.inv(s_r)
        # Compute Atilde
        self.A_tilde = u_r.T @ X2 @ v_r @ s_inv

        
        self.lambda_ , self.W = np.linalg.eig(self.A_tilde)


        self.phi = X2 @ v_r @ s_inv @ self.W

        self.b = np.linalg.pinv(self.phi) @ X1[:,0]
        
        self.omega = np.log(self.lambda_) / self.dt




    # def predict_future(self, t):
    #     pseudophix0 = np.linalg.pinv(self.phi) @ self.X0.reshape(-1, 1)
    #     atphi = self.phi @ np.diag(self.lambda_ ** t)
    #     xt = (atphi @ pseudophix0).reshape(-1)
    #     return xt.real
    
    def predict(self, tvalues):
        time_dynamics = np.zeros((self.r, len(tvalues)), dtype=complex)
        for i, t in enumerate(tvalues):
            time_dynamics[:, i] = self.b * np.exp(self.omega * t)
        xDMD = self.phi @ time_dynamics
        return xDMD.real
    

