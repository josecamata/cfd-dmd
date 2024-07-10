#
# Matlab version incremental SVD algorithm

# https://github.com/bchaoss/incremental-SVD/tree/main

#function [Q,S,R] = InitializeISVD(u1,W)
#     % input: W m*m; u1 m*1, first col of U
#     S = (u1'* W *u1)^(1/2); % Num
#     Q = u1 * S^(-1);        % m*1
#     R = eye(1);
# end

# function [q, V, Q_0, Q,S,R] = UpdateISVD3(q, V, Q_0, Q, S, R, u_l_new, W, tol)
#     % input: Q m*k; S k*k; R l*k;
#     % input: u_l_new m*1,the l+1 col of U; W m*m; tol
#     % input: q cnt; V cell; Q_0;
#     d = Q' * (W * u_l_new);   
#     e = u_l_new - Q*d; p = sqrt(e'* W *e);   
#     if p < tol
#         q = q + 1;
#         V{q} = Q_0' * d;
#     else
#         if q > 0
#             k = size(S,1);
#             Y = [S, cell2mat(V)]; 
#             [Qy, Sy, Ry] = svd(Y, 'econ');
#             Q = Q*(Q_0*Qy); S = Sy;
#             R1 = Ry(1:k,:); R2 = Ry(k+1:end,:); 
#             R = [R*R1; R2];
#             d = Qy' * d;
#         end
#         V = {}; q = 0;
        
#         e = e / p;
#         % Orthogonalization
#         if sqrt(e' * W * Q(:,1)) > tol
#             e = e - Q * (Q' * (W*e)); 
#             p1 = (e'* W *e)^(1/2); e = e / p1;
#         end
        
#         k = size(S,1);
#         Y = [S, Q_0' * d; zeros(1, k), p];
#         [Qy, Sy, Ry] = svd(Y);
#         Q_0 = [Q_0, zeros(size(Q_0, 1), 1); zeros(1, size(Q_0, 2)), 1] * Qy;
#         Q = [Q, e]; S = Sy;
#         R = [R, zeros(size(R, 1), 1); zeros(1, size(R, 2)), 1] * Ry;         
#     end
# end

import numpy as np

class iSVD(object):
    def __init__(self):
        self.u1  = None
        self.W   = None
        self.S   = None
        self.Q   = None
        self.R   = None
        self.q   = 0
        self.Q_0 = 1
        self.V   = []
  
    def Initialize(self, u1, tol=1e-10):
        self.u1 = u1
        self.W  = np.eye(u1.shape[0])
        self.Q_0 = np.ones(u1.shape[0])
        self.S = np.sqrt(self.u1.T @ self.W @ self.u1)
        self.Q = np.zeros((self.u1.shape[0], 1))
        self.Q[:,0] = self.u1 * self.S**(-1)
        self.R = np.eye(1)
        self.tol = tol
        return self.Q, self.S, self.R

    def Update(self, u_l_new):
        d = self.Q.T @ (self.W @ u_l_new)
        e = u_l_new - self.Q * d
        p = np.sqrt(e.T @ self.W @ e)
        if sp < self.tol:
            self.q = self.q + 1
            self.V.append(self.Q_0.T @ d)
        else:
            if self.q > 0:
                k = self.S.shape[1]
                Y = np.concatenate((self.S, np.array(V)), axis=1)
                Qy, Sy, Ry = np.linalg.svd(Y, full_matrices=False)
                self.Q = self.Q @ (self.Q_0 @ Qy)
                S = Sy
                R1 = Ry[0:k, :]
                R2 = Ry[k:, :]
                R = np.concatenate((R @ R1, R2), axis=0)
                d = Qy.T @ d
            self.V = []
            self.q = 0
            e = e / p

            # Orthogonalization
            if np.sqrt(e.T @ self.W @ self.Q[:, 0]) > self.tol:
                e = e - self.Q @ (self.Q.T @ (self.W @ e))
                p1 = np.sqrt(e.T @ self.W @ e)
                e = e / p1
            
            k = self.S.shape[1]
            Y = np.concatenate((self.S, self.Q_0.T @ d), axis=1)
            Y = np.concatenate((Y, np.zeros((1, k))), axis=0)
            Y = np.concatenate((Y, p))
            Qy, Sy, Ry = np.linalg.svd(Y)
            self.Q_0 = np.concatenate((self.Q_0, np.zeros((self.Q_0.shape[0], 1))), axis=1)
            self.Q_0 = np.concatenate((self.Q_0, np.zeros((1, self.Q_0.shape[1]))), axis=0)
            self.Q_0 = self.Q_0 @ Qy
            self.Q = np.concatenate((self.Q, e), axis=1)
            self.S = Sy
            self.R = np.concatenate((self.R, np.zeros((self.R.shape[0], 1))), axis=1)
            self.R = np.concatenate((self.R, np.zeros((1, self.R.shape[1]))), axis=0)
            self.R = self.R @ Ry
        return self.Q, self.S, self.R

# function [Q, S, R] = UpdateISVD3check(q,V,Q_0,Q,S,R)
#     % input: Q m*k; S k*k; R l*k;
#     k = size(S, 1);
#     if q > 0
#         Y = [S, cell2mat(V)];
#         [Qy, Sy, Ry] = svd(Y, 'econ');
        
#         Q = Q*(Q_0*Qy); S = Sy; 
#         R1 = Ry(1:k,:); R2 = Ry(k+1:end,:); 
#         R = [R*R1; R2];
#     else
#         Q = Q * Q_0;
#     end
# end

    def UpdateCheck(self):
        k = self.S.shape[1]
        if self.q > 0:
            Y = np.concatenate((self.S, np.array(self.V)), axis=1)
            Qy, Sy, Ry = np.linalg.svd(Y, full_matrices=False)
            self.Q = self.Q @ (self.Q_0 @ Qy)
            self.S = Sy
            R1 = Ry[0:k, :]
            R2 = Ry[k:, :]
            R = np.concatenate((R @ R1, R2), axis=0)
        else:
            self.Q = self.Q @ self.Q_0
        return self.Q, self.S, self.R
    
