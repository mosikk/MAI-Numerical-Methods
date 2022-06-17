"""
Output:
Least squares method, degree = 1
P(x) = 0.5561647619047624 + 0.631545714285714x
Sum of squared errors = 0.7884327367619048
Least squares method, degree = 2
P(x) = 0.5689421428571434 + 0.6890439285714289x + -0.019166071428571622x^2
Sum of squared errors = 0.7747187737857143
"""

import copy
import matplotlib.pyplot as plt


def LU_decompose(A):
    """
    Stolen from lab 1-1

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
    Stolen from lab 1-1

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


def least_squares(x, y, n):
    """
    Count coefficient of polynom (degree = n) for least squares method for approximating tabular function y = f(x)
    Returns arrays of coeffs
    """
    assert len(x) == len(y)
    A = []
    b = []
    for k in range(n + 1):
        A.append([sum(map(lambda x: x ** (i + k), x)) for i in range(n + 1)])
        b.append(sum(map(lambda x: x[0] * x[1] ** k, zip(y, x))))
    L, U = LU_decompose(A)
    return solve_system(L, U, b)


def P(coefs, x):
    """
    Calculate the value of polynomial function at x
    """
    return sum([c * x**i for i, c in enumerate(coefs)])


def sum_squared_errors(x, y, ls_coefs):
    """
    Calculate sum of squared errors
    """
    y_ls = [P(ls_coefs, x_i) for x_i in x]
    return sum((y_i - y_ls_i)**2 for y_i, y_ls_i in zip(y, y_ls))


if __name__ == '__main__':
    x = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    y = [-0.4597, 1.0, 1.5403, 1.5839, 2.010, 3.3464]
    plt.scatter(x, y, color='r')

    print('Least squares method, degree = 1')
    ls1 = least_squares(x, y, 1)
    print(f'P(x) = {ls1[0]} + {ls1[1]}x')
    plt.plot(x, [P(ls1, x_i) for x_i in x], color='b', label='degree = 1')
    print(f'Sum of squared errors = {sum_squared_errors(x, y, ls1)}')

    print('Least squares method, degree = 2')
    ls2 = least_squares(x, y, 2)
    print(f'P(x) = {ls2[0]} + {ls2[1]}x + {ls2[2]}x^2')
    plt.plot(x, [P(ls2, x_i) for x_i in x], color='g', label='degree = 2')
    print(f'Sum of squared errors = {sum_squared_errors(x, y, ls2)}')

    plt.legend()
    plt.show()
