import numpy as np

def QR(A):
    m , n = A.shape

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        u = A[:, i].copy()
        for j in range(i):
            R[j, i] = Q[:, j] @ A[:, i]
            u = u - R[j , i] * Q[:, j]

        R[i, i] = np.linalg.norm(u)

        if abs(R[i, i]) < 1e-12:
            Q[:, i] = 0
            R[i, i] = 0
            continue
         
        Q[:, i] = u / R[i, i]

    return Q, R

def Cholesky(B):
    n = B.shape[0]
    L = np.zeros((n,n))

    for k in range(n):
        val = B[k, k] - sum(L[k, j] * L[k, j] for j in range(k))
        L[k, k] = np.sqrt(max(val, 1e-14))
        for i in range(k + 1, n):
            L[i, k] = (
                B[i, k] - sum(L[i, j] * L[k, j] for j in range(k))
            ) / L[k,k]
        
    return L

def QR_algorithm(A, max_iter=1000, tol=1e-10):
    n = A.shape[0]
    Ak = A.copy()
    U = np.eye(n)

    for _ in range(max_iter):
        Q , R = QR(Ak)
        Ak = R @ Q
        U = U @ Q
        # det = np.det(Ak)
        off = Ak - np.diag(np.diag(Ak))

        if np.linalg.norm(off) < tol:
            break
    
    eigenvalues = np.diag(Ak)
    eigenvectors = U

    return eigenvalues, eigenvectors



def myeigh(C1, C):
    L = Cholesky(C)
    L_inv = np.linalg.inv(L)

    A = L_inv @ C1 @ L_inv.T

    eigenvalues, eigenvectors_y = QR_algorithm(A)

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors_y = eigenvectors_y[:, idx]

    eigenvectors = L_inv.T @ eigenvectors_y

    return eigenvalues, eigenvectors