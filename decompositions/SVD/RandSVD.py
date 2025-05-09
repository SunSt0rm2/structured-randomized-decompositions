#!/usr/bin/env python
# coding: utf-8

# Run 'setup.py' file before this one using instruction in the beginning

# In[33]:


import sys
print(sys.executable)


# In[34]:

import time
from scipy.linalg import qr # type: ignore
from scipy.linalg import svd # type: ignore
import numpy as np # type: ignore
import transforms # type: ignore

def randomized_svd(A, k, p, f_matmul):

  start = time.time()

  S = np.random.randn(n, k + p)
  Y = f_matmul(A, S)

  print("matmul in rand :", time.time() - start)

  start = time.time()

  Q, _ = qr(Y, mode='economic')

  B = f_matmul(Q.T, A)


  U, s, V = svd(B, full_matrices=False)

  U = f_matmul(Q, U)

  print("qr in rand :", time.time() - start)

  return U, s, V



# In[ ]:





# In[35]:


n = 2**10
m = 2**10
k = 100
p = 10
A = np.zeros((n, m))
for i in range(0, n):
  A[i][i] = 2
  if i > 0:
    A[i - 1][i] = -1
  if i < n - 1:
    A[i + 1][i] = -1


#s_rand = randomized_svd(A, k, p)
#print(s_rand)


# In[36]:


def my_matmul(A, B):
  n, m = A.shape
  _, k = B.shape
  C = np.zeros((n, k))
  for i in range(0, n):
    for j in range(0, k):
      for l in range(0, m):
        C[i][j] += A[i][l] * B[l][j]
  return C


# In[37]:


def matmul(A, B):
  return A @ B


# 

# In[38]:


import numpy as np # type: ignore
from scipy.linalg import svd # type: ignore

def fast_hadamard_transform(x):
    n = len(x)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, min(i + h, n - h)):
                  x_j = x[j]
                  x[j] = x_j + x[j + h]
                  x[j + h] = x_j - x[j + h]
        h *= 2
    return x / np.sqrt(n)

def hadamard_svd(A, k, f_matmul, p=5, random = True):
    m, n = A.shape
    power = 1
    l = k + p
    start = time.time()
    if random:
        Omega = np.random.randn(n, l)
        for i in range(0, n):
          Omega[i, :] = transforms.fast_hadamard_transform(Omega[i, :])
        Y = f_matmul(A, Omega)
    else:
        Y = A
        for i in range(0, m):
          Y[i, :] = transforms.fast_hadamard_transform(Y[i, :])
        random_cols = np.random.choice(n, l, replace = False)
        Y = Y[:, random_cols]

    print("hadamar calc:", time.time() - start)

    start = time.time()

    Q, _ = qr(Y, mode='economic')

    B = f_matmul(Q.T, A)

    U, s, Vt = svd(B, full_matrices=False)

    U = f_matmul(Q, U)

    print("qr in hadamar :", time.time() - start)

    return U, s, Vt[:, :n]


# 

# In[39]:


def format_from_svd(U, s, V):
  n, m = U.shape
  full = np.zeros(m)
  full[:len(s)] = s
  S = np.diag(full)
  return U @ S @ V


# In[40]:


import time


# In[41]:


#A = np.random.randn(n, m)
v = np.random.randn(n, k+p)
w = np.random.randn(k+p, m)
A = v @ w

import time
import numpy as np # type: ignore
from scipy.linalg import svd # type: ignore

start = time.time()
U_h_r, s_hadamar_r, V_h_r = hadamard_svd(A, k, matmul, p, random = False)
print(time.time() - start)

start = time.time()
U_h, s_hadamar, V_h = hadamard_svd(A, k, matmul, p, random = True)
print(time.time() - start)

start = time.time()
U_r, s_rand, V_r = randomized_svd(A, k, p, matmul)
print(time.time() - start)

# start = time.time()
# U, s, V = np.linalg.svd(A)
# print(time.time() - start)

# start = time.time()
# U_h_r, s_hadamar_r, V_h_r = hadamard_svd(A, k, my_matmul, p, random = False)
# print(time.time() - start)

# start = time.time()
# U_r, s_rand, V_r = randomized_svd(A, k, p, my_matmul)
# print(time.time() - start)


# In[42]:


#print(np.linalg.norm(format_from_svd(U, s, V) - A))
print(np.linalg.norm(format_from_svd(U_h, s_hadamar, V_h) - A))
print(np.linalg.norm(format_from_svd(U_h_r, s_hadamar_r, V_h_r) - A))
print(np.linalg.norm(format_from_svd(U_r, s_rand, V_r) - A))


# In[43]:


def fft(x):
  n = len(x)
  if n == 1:
      return x
  even = fft(x[::2])
  odd = fft(x[1::2])
  rotate = np.exp(-2 * np.pi * np.arange(n // 2) / n)
  first_half = even + rotate * odd
  second_half = even - rotate * odd
  return np.concatenate([first_half, second_half])


# In[44]:


def fft_svd(A, k, f_matmul, p=5):
  m, n = A.shape
  Y = A
  l = k + p

  start = time.time()
  for i in range(0, m):
    Y[i, :] = transforms.fft(Y[i, :])

  random_cols = np.random.choice(n, l, replace = False)
  Y = Y[:, random_cols]
  print("fft calc:", time.time() - start)
  start = time.time()

  Q, _ = qr(Y, mode='economic')

  B = f_matmul(Q.T, A)

  U, s, Vt = svd(B, full_matrices=False)

  U = f_matmul(Q, U)

  print("qr in fft :", time.time() - start)

  return U, s, Vt[:, :n]



# In[45]:


start = time.time()
U_fft, s_fft, V_fft = fft_svd(A, k, matmul, p)
print(time.time() - start)
print(np.linalg.norm(format_from_svd(U_fft, s_fft, V_fft) - A))


# In[46]:


def export_projection(A, k, f_matmul, p=5, random = True):
    m, n = A.shape
    power = 1
    l = k + p
    start = time.time()
    if random:
        Omega = np.random.randn(n, l)
        for i in range(0, n):
          Omega[i, :] = transforms.fast_hadamard_transform(Omega[i, :])
        Y = f_matmul(A, Omega)
    else:
        Y = A
        for i in range(0, m):
          Y[i, :] = transforms.fast_hadamard_transform(Y[i, :])
        random_cols = np.random.choice(n, l, replace = False)
        Y = Y[:, random_cols]
    return Y

