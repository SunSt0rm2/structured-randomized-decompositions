import numpy as np
import math
import typing as tp


def calculateOrthoMatrix(matrix, eps, n, m, r) -> tp.Any:
    w = np.zeros((r + r, n))
    w[:r] = np.random.randn(r, n)
    y = np.zeros((r + r, m))
    for i in range(r):
        y[i] = matrix @ w[i]
    j = 0
    Q = np.zeros((m, r))
    while np.max(np.linalg.norm(y[j + 1: j + r], axis=1)) > eps / (10 * math.sqrt(2 / math.pi)):
        j = j + 1
        y[j] = (np.eye(m) - Q[:, :j-1] @ Q[:, :j - 1].T) @ y[j]
        q = y[j] / np.linalg.norm(y[j])
        Q[:, j - 1] = q
        w[j + r] = np.random.randn(n)
        y[j + r] = (np.eye(m) - Q[:, :j] @ Q[:, :j].T) @ (matrix @ w[j + r])
        for i in range(j + 1, j + r):
            y[i] = y[i] - q.reshape(-1, 1) @ (q.reshape(1, -1) @ y[i].reshape(-1, 1)).reshape(-1)

    return Q[:, :j]

def calculateOrthTim(A, eps, n, m,r):
    j = 0
    Y = np.random.rand(m, r)
    Y = A @ Y
    Q = Y[:, 0].reshape(-1,1)/np.linalg.norm(Y[:, 0])
    while np.max(np.linalg.norm(Y[:, j + 1 : j + r], axis = 0)) > eps / (10 * math.sqrt(2 / math.pi)):
        j = j + 1
        Y[:, j] = (np.eye(n) - Q @ Q.T) @ Y[:, j]
        q = Y[:, j].reshape(-1,1) / np.linalg.norm(Y[:, j])
        Q = np.concatenate((Q, q), axis = 1)
        y_new = A @ np.random.randn(m, 1)
        y_new = (np.eye(n) - Q @ Q.T) @ y_new.reshape(-1,1)
        Y = np.concatenate((Y, y_new), axis = 1)
        for i in range(j + 1, j + r):
            Y[:, i] = Y[:, i] - q @ (q.T @ Y[:, i])
    return Q



def checkOrtho(matrix, axis) -> tp.Any:
    if axis == 0:
        prod = matrix.T @ matrix
    else:
        prod = matrix @ matrix.T
    return np.allclose(prod, np.eye(prod.shape[0]))


n = 100
m = 75
r = 10
A = np.random.randn(m, n)

