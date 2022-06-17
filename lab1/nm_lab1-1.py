"""
For input:
-6 -5 -3 -8
5 -1 -5 -4
-6 0 5 5
-7 -2 8 5

101 51 -53 -63


Output:
LU decomposition
L:
  1.00   0.00   0.00   0.00
 -0.83   1.00   0.00   0.00
  1.00  -0.97   1.00   0.00
  1.17  -0.74   8.00   1.00
U:
 -6.00  -5.00  -3.00  -8.00
  0.00  -5.17  -7.50 -10.67
  0.00   0.00   0.74   2.68
  0.00  -0.00   0.00 -15.00
System solution
x: [-2.0, -3.0000000000000053, -6.000000000000004, -6.999999999999996]
det A = -344.99999999999994
A^(-1)
 -0.07   0.07  -0.20   0.14
  0.01  -0.61  -0.26  -0.21
 -0.02  -0.38  -0.58   0.24
 -0.07   0.47   0.53  -0.07
"""

import copy


def LU_decompose(A):
    """
    A = LU, where:
    L - lower triangle matrix
    U - upper triangle matrix
    Returns L, U
    """
    n = len(A)
    L = [[0 for _ in range(n)] for _ in range(n)]
    U = copy.deepcopy(A)

    for k in range(1, n):
        for i in range(k - 1, n):
            for j in range(i, n):
                L[j][i] = U[j][i] / U[i][i]

        for i in range(k, n):
            for j in range(k - 1, n):
                U[i][j] = U[i][j] - L[i][k - 1] * U[k - 1][j]

    return L, U


def solve_system(L, U, b):
    """
    Solves system of equations: LUx=b
    Returns x
    """
    # Step 1: Ly = b
    n = len(L)
    y = [0 for _ in range(n)]
    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = (b[i] - s) / L[i][i]

    # Step 2: Ux = y
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(n - 1, i - 1, -1):
            s += U[i][j] * x[j]
        x[i] = (y[i] - s) / U[i][i]
    return x


def determinant(A):
    """
    Calculate the determinant of matrix A
    """
    _, U = LU_decompose(A)
    det = 1
    for i in range(len(U)):
        det *= U[i][i]
    return det


def inverse_matrix(A):
    """
    Calculate A^(-1)
    """
    n = len(A)
    E = [[float(i ==j) for i in range(n)] for j in range(n)]
    L, U = LU_decompose(A)
    A_inv = []
    for e in E:
        inv_row = solve_system(L, U, e)
        A_inv.append(inv_row)
    return transpose(A_inv)


def mat_mult(X, Y):
    # X = (m x p)
    # Y = (p x n)
    # R = XY = (m x n)
    m = len(X)
    p = len(Y)
    n = len(Y[0])
    R = [[0 for _ in range(n)] for _ in range(m)]

    for i in range(m):
        for j in range(n):
            for k in range(p):
                R[i][j] += X[i][k] * Y[k][j]
    return R


def transpose(X):
    m = len(X)
    n = len(X[0])
    X_T = [[X[j][i] for j in range(n)] for i in range(m)]
    return X_T


def print_matrix(A):
    m = len(A)
    n = len(A[0])
    for i in range(m):
        for j in range(n):
            print(f'%6.2f' % A[i][j], end=' ')
        print()


if __name__ == '__main__':
    n = int(input('Enter the size of matrix: '))

    print('Enter matrix A')
    A = [[] for _ in range(n)]
    for i in range(n):
        row = list(map(int, input().split()))
        A[i] = row
    print('Enter vector b')
    b = list(map(int, input().split()))

    print("LU decomposition")
    L, U = LU_decompose(A)
    print('L:')
    print_matrix(L)
    print('U:')
    print_matrix(U)

    print("System solution")
    x = solve_system(L, U, b)
    print('x:', x)

    print("det A =", determinant(A))

    print("A^(-1)")
    print_matrix(inverse_matrix(A))