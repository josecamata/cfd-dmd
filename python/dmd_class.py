import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

class DMD(object):
    def __init__(self, r=2, svd_rank=0):
        self.r = r

    def fit(self, X):

        N,T = X.shape

        X1 = X[:, :-1] 
        X2 = X[:, 1 :]
        # Compute SVD of x (uu,ss,vv)
        u,s,v = np.linalg.svd(X1,full_matrices=False)
        u_r=u[: , :self.r]
        s_r=s[:self.r]
        v_r=v.conj().T[:,:self.r]

        s_inv = np.reciprocal(s_r)

        # Compute Atilde
        A_tilde = u_r.conj().T @ X2 @ v_r * s_inv
       
        eigenvalues,eigenvectors = np.linalg.eig(A_tilde)
        
        # Reconstruct DMD modes (phi)
        self.phi          = X2 @ v_r @ np.diag(s_inv) @ eigenvectors 
        self.eigenvalues  = eigenvalues
        self.eigenvectors = eigenvectors
        self.A            = self.phi @ np.diag(self.eigenvalues) @ np.linalg.pinv(self.phi)

        # lambda = np.diag(self.eigenvalues)

        self.omega  = np.diag(np.log(self.eigenvalues))

        # # Compute time evolution of DMD modes (b)
        self.b = np.linalg.pinv(self.phi) @ X1[:,0]

        #
        # TODO: check how to compute time dynamics
        # time_dynamics=np.zeros([self.r,T-1], dtype='complex')
        # for i in range(T-1):
        #     time_dynamics[:,i]=np.multiply(np.exp(self.omega*i),self.b)

    def predict(self, X, t):
        # Compute DMD reconstruction
        return self.phi @ (self.b * np.exp(self.omega * t))



