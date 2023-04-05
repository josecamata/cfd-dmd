import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

class DMD(object):
    def __init__(self, r=2, svd_rank=0):
        self.r = r

    def fit(self, X):

        n_points,n_snapshots = X.shape

        self.snapshots_shape = X.shape

        X1 = X[:,:-1] 
        X2 = X[:, 1:]

        # Compute SVD of x (uu,ss,vv)
        u,s,v = np.linalg.svd(X1,full_matrices=False)
        u_r   = u[: , :self.r]
        s_r   = s[:self.r]
        v_r   = v.conj().T[:,:self.r]

        s_inv = np.diag(np.reciprocal(s_r))

        # Compute Atilde
        A_tilde = u_r.conj().T @ X2 @ v_r @ s_inv
       
        eigenvalues,eigenvectors = np.linalg.eig(A_tilde)
        
        # Reconstruct DMD modes (phi)
        self.phi          = X2 @ v_r @ s_inv @ eigenvectors 
        self.eigenvalues  = eigenvalues
        self.eigenvectors = eigenvectors
        self.A            = np.linalg.multi_dot([self.phi ,np.diag(self.eigenvalues), np.linalg.pinv(self.phi)])

        # lambda = np.diag(self.eigenvalues)

        self.omega  = np.diag(np.log(self.eigenvalues))

        # # Compute time evolution of DMD modes (b)
        self.b = np.linalg.pinv(self.phi) @ X1[:,0]

    def predict(self, X):
        return np.dot(self.A, X)
    

    def get_n_snapshots(self):
        if self.snapshots_shape is None:
            raise ValueError("snapshots_shape is None. Please fit the model first.")
        return self.snapshots_shape[1]
