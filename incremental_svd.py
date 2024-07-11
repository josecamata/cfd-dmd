import numpy as np
import time
from scipy.linalg import svd

def InitializeISVD(u1, W):
    S = np.sqrt(u1.T @ W @ u1)
    Q = u1 / S
    R = np.eye(1)
    return Q, S, R

def UpdateISVD3(q, V, Q_0, Q, S, R, u_l_new, W, tol):
    d = Q.T @ (W @ u_l_new)
    e = u_l_new - Q @ d
    p = np.sqrt(e.T @ W @ e)
    
    if p < tol:
        q += 1
        V.append(Q_0.T @ d)
    else:
        if q > 0:
            k = S.shape[0]
            Y = np.hstack([S, np.column_stack(V)])
            Qy, Sy, Ry = svd(Y, full_matrices=False)
            Q = Q @ (Q_0 @ Qy)
            S = Sy
            R1 = Ry[:k, :]
            R2 = Ry[k:, :]
            R = np.vstack([R @ R1, R2])
            d = Qy.T @ d
        
        V = []
        q = 0
        e /= p
        
        if np.sqrt(e.T @ W @ Q[:, 0]) > tol:
            e -= Q @ (Q.T @ (W @ e))
            p1 = np.sqrt(e.T @ W @ e)
            e /= p1
        
        k = S.shape[0]
        Y = np.vstack([np.hstack([S, Q_0.T @ d]), np.zeros((1, k + 1))])
        Y[-1, -1] = p
        Qy, Sy, Ry = svd(Y)
        Q_0 = np.vstack([np.hstack([Q_0, np.zeros((Q_0.shape[0], 1))]), np.zeros((1, Q_0.shape[1] + 1))]) @ Qy
        Q = np.column_stack([Q, e])
        S = Sy
        R = np.vstack([np.hstack([R, np.zeros((R.shape[0], 1))]), np.zeros((1, R.shape[1] + 1))]) @ Ry
    
    return q, V, Q_0, Q, S, R

def UpdateISVD3check(q, V, Q_0, Q, S, R):
    k = S.shape[0]
    if q > 0:
        Y = np.hstack([S, np.column_stack(V)])
        Qy, Sy, Ry = svd(Y, full_matrices=False)
        Q = Q @ (Q_0 @ Qy)
        S = Sy
        R1 = Ry[:k, :]
        R2 = Ry[k:, :]
        R = np.vstack([R @ R1, R2])
    else:
        Q = Q @ Q_0
    
    return Q, S, R

# Parâmetros
tol = 1e-15

# Gerando a matriz U
U = np.random.rand(100, 30) @ np.random.rand(30, 80)
m = U.shape[0]
W = np.eye(m)
u1 = U[:, 0]

# ISVD
start = time.time()
Q, S, R = InitializeISVD(u1, W)
V = []
Q_0 = 1
q = 0
n = U.shape[1]

for L in range(1, n):
    u_l_new = U[:, L]
    q, V, Q_0, Q, S, R = UpdateISVD3(q, V, Q_0, Q, S, R, u_l_new, W, tol)

Q, S, R = UpdateISVD3check(q, V, Q_0, Q, S, R)
end = time.time()
print(f'Tempo ISVD: {end - start} segundos')

# SVD padrão
start = time.time()
Q_st, S_st, R_st = svd(U, full_matrices=False)
end = time.time()
print(f'Tempo SVD: {end - start} segundos')

# Comparação de normas
print('Norma (Q*S*R.T - U):', np.linalg.norm(Q @ S @ R.T - U))
print('Norma (Q_st*S_st*R_st.T - U):', np.linalg.norm(Q_st @ np.diag(S_st) @ R_st.T - U))
print('Norma (abs(Q[:, -1].T @ W @ Q[:, 0])):', np.linalg.norm(np.abs(Q[:, -1].T @ W @ Q[:, 0])))