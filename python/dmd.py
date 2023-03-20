import numpy as np

def DMD(data, r):
    """Dynamic Mode Decomposition (DMD) algoritmo."""
    
    # Constrói matrizes de dados.
    X1 = data[:, : -1]
    X2 = data[:, 1 :]

    # Realiza a decomposição singular de X1.
    u, s, v = np.linalg.svd(X1, full_matrices = False)

    # Determina a matriz de Koopman.
    A_tilde = u[:, : r].conj().T @ X2 @ v[: r, :].conj().T * np.reciprocal(s[: r])

    # Calcula os autovalores e autovetores da matriz de Koopman.
    Phi, Q = np.linalg.eig(A_tilde)

    # Calcula a matriz de coeficientes.
    Psi = X2 @ v[: r, :].conj().T @ np.diag(np.reciprocal(s[: r])) @ Q
    A = Psi @ np.diag(Phi) @ np.linalg.pinv(Psi)
    
    return A_tilde, Phi, A

def DMD4cast(data, r, pred_step):
    N, T = data.shape
    _, _, A = DMD(data, r)      # É retornado a matriz de Koopman, autovalores e matriz de coeficientes.
    mat = np.append(data, np.zeros((N, pred_step)), axis = 1)
    for s in range(pred_step):      # Preenche a matriz mat com os valores previstos.
        mat[:, T + s] = (A @ mat[:, T + s - 1]).real
    return mat[:, - pred_step :]

X = np.zeros((2, 10))       # Gera matrizes preenchidas com zeros.
X[0, :] = np.arange(1, 11)      # Retorna valores uniformemente espaçados dentro do intervalor [..., ...)
X[1, :] = np.arange(2, 12)      # e substitui os zeros dentro das matrizes.
pred_step = 2
r = 2
mat_hat = DMD4cast(X, r, pred_step)
print(mat_hat)
