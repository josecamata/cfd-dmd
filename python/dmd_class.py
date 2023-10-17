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

    def fit(self, X, thresh=0.7,dt=1.0):

        self.dt = dt
        self.n_points,self.n_snapshots = X.shape

        X1 = X[:,:-1] 
        X2 = X[:, 1:]
        self.X0 = X1[:,0]

        # Compute SVD of x (uu,ss,vv)
        u,s,v = np.linalg.svd(X1)
        q = np.cumsum(s) / np.sum(s)
        mask = q > thresh
        self.r = 2 # np.where(mask)[0][0]

        u_r   = u[: , :self.r].conj()
        s_r   = np.diag(s[:self.r])
        v_r   = v[:self.r:, ].conj().transpose()

        #np.disp(u_r.real);
        #np.disp(s_r.real);   
        #np.disp(v_r.real);

        s_inv = np.linalg.inv(s_r)
        # Compute Atilde
        self.A_tilde = u_r.T @ X2 @ v_r @ s_inv

        # np.disp(self.A_tilde.real)
        
        self.lambda_ , self.W = np.linalg.eig(self.A_tilde)

        # np.disp(self.lambda_.real)
        # np.disp(self.W.real)

        self.phi = X2 @ v_r @ s_inv @ self.W

        # np.disp(self.phi.real)

        self.b = np.linalg.pinv(self.phi) @ X1[:,0]
        self.omega = np.log(self.lambda_) / self.dt

        np.disp(self.omega.real)


    def predict_future(self, t):
        pseudophix0 = np.linalg.pinv(self.phi) @ self.X0.reshape(-1, 1)
        atphi = self.phi @ np.diag(self.lambda_ ** t)
        xt = (atphi @ pseudophix0).reshape(-1)
        return xt.real
    
    def predict(self, tvalues):
        time_dynamics = np.zeros((self.r, len(tvalues)), dtype=np.complex)
        for i, t in enumerate(tvalues):
            time_dynamics[:, i] = self.b * np.exp(self.omega * t)
        xDMD = self.phi @ time_dynamics
        return xDMD
    

