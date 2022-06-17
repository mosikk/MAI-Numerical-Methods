"""
For input:

-1 2 9
9 3 4
8 -4 -6

Output:
Eigen values: [-13.007048150840406, (4.503141081268443+2.803623878520362j), (4.503141081268443-2.803623878520362j)]
"""

import numpy as np


def sign(x):
    return -1 if x < 0 else 1 if x > 0 else 0


def L2_norm(vec):
    """
    Counts L2 norm of a vector
    """
    ans = 0
    for num in vec:
        ans += num * num
    return np.sqrt(ans)


def get_householder_matrix(A, col_num):
    """
    Get householder matrix for iteration of QR decomposition
    Returns householder matrix H
    """
    n = A.shape[0]
    v = np.zeros(n)
    a = A[:, col_num]
    v[col_num] = a[col_num] + sign(a[col_num]) * L2_norm(a[col_num:])
    for i in range(col_num + 1, n):
        v[i] = a[i]
    v = v[:, np.newaxis]
    H = np.eye(n) - (2 / (v.T @ v)) * (v @ v.T)
    return H


def QR_decomposition(A):
    """
    Make QR decomposition: A = QR
    Returns Q, R
    """
    n = A.shape[0]
    Q = np.eye(n)
    A_i = np.copy(A)

    for i in range(n - 1):
        H = get_householder_matrix(A_i, i)
        Q = Q @ H
        A_i = H @ A_i
    return Q, A_i


def get_roots(A, i):
    """
    Get roots of system of two equations (i and i+1) of matrix A
    """
    n = A.shape[0]
    a11 = A[i][i]
    a12 = A[i][i + 1] if i + 1 < n else 0
    a21 = A[i + 1][i] if i + 1 < n else 0
    a22 = A[i + 1][i + 1] if i + 1 < n else 0
    return np.roots((1, -a11 - a22, a11 * a22 - a12 * a21))


def is_complex(A, i, eps):
    """
    Check if i and (i+1)-th eigen values are complex
    """
    Q, R = QR_decomposition(A)
    A_next = np.dot(R, Q)
    lambda1 = get_roots(A, i)
    lambda2 = get_roots(A_next, i)
    return abs(lambda1[0] - lambda2[0]) <= eps and abs(lambda1[1] - lambda2[1]) <= eps


def get_eigen_value(A, i, eps):
    """
    Get i-th (and (i+1)-th if complex) eigen value of matrix A.
    Returns eigen value(s) and matrix A_i for the next iterations
    """
    A_i = np.copy(A)
    while True:
        Q, R = QR_decomposition(A_i)
        A_i = R @ Q
        if L2_norm(A_i[i + 1:, i]) <= eps:
            return A_i[i][i], A_i
        elif L2_norm(A_i[i + 2:, i]) <= eps and is_complex(A_i, i, eps):
            return get_roots(A_i, i), A_i


def get_eigen_values_QR(A, eps):
    """
    Count all eigen values of A using QR decomposition
    """
    n = A.shape[0]
    A_i = np.copy(A)
    eigen_values = []

    i = 0
    while i < n:
        cur_eigen_values, A_i_plus_1 = get_eigen_value(A_i, i, eps)
        if isinstance(cur_eigen_values, np.ndarray):
            # complex
            eigen_values.extend(cur_eigen_values)
            i += 2
        else:
            # real
            eigen_values.append(cur_eigen_values)
            i += 1
        A_i = A_i_plus_1
    return eigen_values


if __name__ == '__main__':
    n = int(input('Enter the size of matrix: '))

    print('Enter matrix A')
    A = [[] for _ in range(n)]
    for i in range(n):
        row = list(map(int, input().split()))
        A[i] = row
    A = np.array(A, dtype='float')
    eps = float(input('Enter epsilon: '))

    eig_values = get_eigen_values_QR(A, eps)
    print('Eigen values:', eig_values)
