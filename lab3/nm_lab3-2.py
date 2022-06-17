"""
Output:
[0.0; 1.0)
s(x) = 1.0 + 0.6844107142857142(x - 0.0) + 0(x - 0.0)^2 + -0.14411071428571423(x - 0.0)^3
[1.0; 2.0)
s(x) = 1.5403 + 0.25207857142857154(x - 1.0) + -0.4323321428571427(x - 1.0)^2 + 0.22385357142857124(x - 1.0)^3
[2.0; 3.0)
s(x) = 1.5839 + 0.05897499999999989(x - 2.0) + 0.239228571428571(x - 2.0)^2 + 0.12789642857142888(x - 2.0)^3
[3.0; 4.0)
s(x) = 2.01 + 0.9211214285714284(x - 3.0) + 0.6229178571428576(x - 3.0)^2 + -0.2076392857142859(x - 3.0)^3
s(x_test) = s(1.5) = 1.6891816964285713
"""


import numpy as np
import matplotlib.pyplot as plt


def s(a, b, c, d, x):
    """
    Calculate s(x)
    """
    return a + b * x + c * x**2 + d * x**3


def tridiagonal_solve(A, b):
    """
    From lab 1-2
    Solves Ax=b, where A - tridiagonal matrix
    Returns x
    """
    n = len(A)
    # Step 1. Forward
    v = [0 for _ in range(n)]
    u = [0 for _ in range(n)]
    v[0] = A[0][1] / -A[0][0]
    u[0] = b[0] / A[0][0]
    for i in range(1, n-1):
        v[i] = A[i][i+1] / (-A[i][i] - A[i][i-1] * v[i-1])
        u[i] = (A[i][i-1] * u[i-1] - b[i]) / (-A[i][i] - A[i][i-1] * v[i-1])
    v[n-1] = 0
    u[n-1] = (A[n-1][n-2] * u[n-2] - b[n-1]) / (-A[n-1][n-1] - A[n-1][n-2] * v[n-2])

    # Step 2. Backward
    x = [0 for _ in range(n)]
    x[n-1] = u[n-1]
    for i in range(n-1, 0, -1):
        x[i-1] = v[i-1] * x[i] + u[i-1]
    return x


def spline_interpolation(x, y, x_test):
    """
    Get cubic spline interpolation s(x) of tabular function y = f(x)
    s(x) = a + b(x - x_{i-1}) + c(x - x_{i-1})^2 + d(x - x_{i-1})^3
    Returns:
        arrays of coefficients a, b, c, d for each interval,
        s(x_test)
    """
    assert len(x) == len(y)
    n = len(x)

    # Step 1. Get c coefs
    h = [x[i] - x[i - 1] for i in range(1, len(x))]
    # tridiagonal matrix for calculating c:
    A = [[0 for _ in range(len(h)-1)] for _ in range(len(h)-1)]
    A[0][0] = 2 * (h[0] + h[1])
    A[0][1] = h[1]
    for i in range(1, len(A) - 1):
        A[i][i-1] = h[i-1]
        A[i][i] = 2 * (h[i-1] + h[i])
        A[i][i+1] = h[i]
    A[-1][-2] = h[-2]
    A[-1][-1] = 2 * (h[-2] + h[-1])

    m = [3.0 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1]) for i in range(1, len(h))]

    c = [0] + tridiagonal_solve(A, m)

    # Step 2. Get a coefs
    a = [y[i-1] for i in range(1, n)]

    # Step 3. Get b coefs
    b = [(y[i] - y[i-1]) / h[i-1] - (h[i-1] / 3.0) * (2.0 * c[i-1] + c[i]) for i in range(1, len(h))]
    b.append((y[-1] - y[-2]) / h[-1] - (2.0 * h[-1] * c[-1]) / 3.0)

    # Step 4. Get d coefs
    d = [(c[i] - c[i-1]) / (3.0 * h[i-1]) for i in range(1, len(h))]
    d.append(-c[-1] / (3.0 * h[-1]))

    # Calculate s(x_test)
    for interval in range(len(x)):
        if x[interval] <= x_test < x[interval+1]:
            i = interval
            break
    y_test = s(a[i + 1], b[i + 1], c[i + 1], d[i + 1], x_test - x[i])
    return a, b, c, d, y_test


def draw_plot(x_original, y_original, a, b, c, d):
    """
    Plot spline and original points
    """
    x, y = [], []
    for i in range(len(x_original) - 1):
        x1 = np.linspace(x_original[i], x_original[i + 1], 10)
        y1 = [s(a[i], b[i], c[i], d[i], j - x_original[i]) for j in x1]
        x.append(x1)
        y.append(y1)

    plt.scatter(x_original, y_original, color='r')
    for i in range(len(x_original) - 1):
        plt.plot(x[i], y[i], color='b')
    plt.show()


if __name__ == '__main__':
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [1.0, 1.5403, 1.5839, 2.01, 3.3464]
    x_test = 1.5

    a, b, c, d, y_test = spline_interpolation(x, y, x_test)
    for i in range(len(x) - 1):
        print(f'[{x[i]}; {x[i+1]})')
        print(f's(x) = {a[i]} + {b[i]}(x - {x[i]}) + {c[i]}(x - {x[i]})^2 + {d[i]}(x - {x[i]})^3')
    print(f's(x_test) = s({x_test}) = {y_test}')
    draw_plot(x, y, a, b, c, d)
