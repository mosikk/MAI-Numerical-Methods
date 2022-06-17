"""
Output:
Iterations
Step = 0.1
shooting: 2
Step = 0.05
shooting: 2
Mean absolute errors
Step = 0.1
shooting: 9.494474130121822e-05
finite difference: 0.00015496749798701607
Step = 0.05
shooting: 9.422480418318026e-05
finite difference: 0.00010998854411740988
Runge-Romberg accuracy
shooting: 2.547224856404316e-07
finite difference: 1.0191399734451214e-05
"""


from math import exp
import numpy as np
import matplotlib.pyplot as plt


def runge_kutta_method(f, g, y0, z0, interval, h, return_z=False):
    """
    Stolen from lab 4-1
    Solve cauchy problem:
    y' = g(x, y, z)
    z' = f(x, y, z)
    y(0) = y0
    z(0) = z0
    where x in interval, h - step using runge-kutta method
    Returns solution - tabular function y=f(x) as two arrays: x, y
    If return_z is true, function will also return coefficients for z
    """
    l, r = interval
    x = [i for i in np.arange(l, r + h, h)]
    y = [y0]
    z = [z0]
    for i in range(len(x) - 1):
        K1 = h * g(x[i], y[i], z[i])
        L1 = h * f(x[i], y[i], z[i])
        K2 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K1, z[i] + 0.5 * L1)
        L2 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K1, z[i] + 0.5 * L1)
        K3 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K2, z[i] + 0.5 * L2)
        L3 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K2, z[i] + 0.5 * L2)
        K4 = h * g(x[i] + h, y[i] + K3, z[i] + L3)
        L4 = h * f(x[i] + h, y[i] + K3, z[i] + L3)
        delta_y = (K1 + 2 * K2 + 2 * K3 + K4) / 6
        delta_z = (L1 + 2 * L2 + 2 * L3 + L4) / 6
        y.append(y[i] + delta_y)
        z.append(z[i] + delta_z)

    if not return_z:
        return x, y
    else:
        return x, y, z


def tridiagonal_solve(A, b):
    """
    Stolen from lab 1-2
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


def f(x, y, z):
    return (2 * z + exp(x) * y) / (exp(x) + 1)


def g(x, y, z):
    return z


# Functions for finite difference method
# y'' + p_fd(x)y' + q_fd(x)y = f_fd(x)

def p_fd(x):
    return -2 / (exp(x) + 1)


def q_fd(x):
    return -exp(x) / (exp(x) + 1)


def f_fd(x):
    return 0


def exact_solution(x):
    """
    Exact analytical solution of given problem
    """
    return exp(x) - 1 + 1 / (exp(x) + 1)


def shooting_method(f, g, y0, yn, interval, h, eps):
    """
    Solve boundary value problem:
    y' = g(x, y, z)
    z' = f(x, y, z)
    y(0) = y0
    y(n) = yn
    where x in interval, h - step using shooting method
    Returns:
        solution - tabular function y=f(x) as two arrays: x, y,
        number of iterations
    """
    n_prev = 1.0
    n = 0.8
    iterations = 0
    while True:
        iterations += 1
        x_prev, y_prev = runge_kutta_method(f, g, y0, n_prev, interval, h)
        x, y = runge_kutta_method(f, g, y0, n, interval, h)
        if abs(y[-1] - yn) < eps:
            break
        n_prev, n = n, n - (y[-1] - yn) * (n - n_prev) / ((y[-1] - yn) - (y_prev[-1] - yn))
    return x, y, iterations


def finite_difference_method(p, q, f, y0, yn, interval, h):
    """
    Solve boundary value problem:
    y'' + p(x)y' + q(x)y = f(x)
    y(a) = y0
    y(b) = yn
    where x in interval, h - step using finite difference method
    Returns solution - tabular function y=f(x) as two arrays: x, y,
    """
    A = []
    B = []
    rows = []
    a, b = interval
    x = np.arange(a, b + h, h)
    n = len(x)

    # Creating tridiagonal matrix
    for i in range(n):
        if i == 0:
            rows.append(1)
        else:
            rows.append(0)
    A.append(rows)
    B.append(y0)

    for i in range(1, n - 1):
        rows = []
        B.append(f(x[i]))
        for j in range(n):
            if j == i - 1:
                rows.append(1 / h ** 2 - p(x[i]) / (2 * h))
            elif j == i:
                rows.append(-2 / h ** 2 + q(x[i]))
            elif j == i + 1:
                rows.append(1 / h ** 2 + p(x[i]) / (2 * h))
            else:
                rows.append(0)
        A.append(rows)

    rows = []
    B.append(yn)
    for i in range(n):
        if i == n - 1:
            rows.append(1)
        else:
            rows.append(0)

    A.append(rows)
    y = tridiagonal_solve(A, B)
    return x, y


def runge_romberg_method(h1, h2, y1, y2, p):
    """
    Find more accuracy of solution using previous calculations.
    Works if h1 == 2 * h2
    """
    assert h1 == h2 * 2
    norm = 0
    for i in range(len(y1)):
        norm += (y1[i] - y2[i * 2]) ** 2
    return norm ** 0.5 / (2 ** p + 1)


def mae(y1, y2):
    """
    Find mean absolute error between y1 and y2
    MAE = 1/n sum(|y1-y2|)
    """
    assert len(y1) == len(y2)
    res = 0
    for i in range(len(y1)):
        res += abs(y1[i] - y2[i])
    return res / len(y1)


if __name__ == '__main__':
    y0 = 0.5
    y1 = 1.987
    interval = (0, 1)  # x in [0; 1]
    h = 0.1
    eps = 0.001

    x_shooting, y_shooting, iters_shooting = shooting_method(f, g, y0, y1, interval, h, eps)
    plt.plot(x_shooting, y_shooting, label=f'shooting method, step={h}')
    x_shooting2, y_shooting2, iters_shooting2 = shooting_method(f, g, y0, y1, interval, h / 2, eps)
    plt.plot(x_shooting2, y_shooting2, label=f'shooting method, step={h / 2}')

    x_fd, y_fd = finite_difference_method(p_fd, q_fd, f_fd, y0, y1, interval, h)
    plt.plot(x_fd, y_fd, label=f'finite difference method, step={h}')
    x_fd2, y_fd2 = finite_difference_method(p_fd, q_fd, f_fd, y0, y1, interval, h / 2)
    plt.plot(x_fd2, y_fd2, label=f'finite difference method, step={h / 2}')

    x_exact = [i for i in np.arange(interval[0], interval[1] + h, h)]
    x_exact2 = [i for i in np.arange(interval[0], interval[1] + h / 2, h / 2)]
    y_exact = [exact_solution(x_i) for x_i in x_exact]
    y_exact2 = [exact_solution(x_i) for x_i in x_exact2]
    plt.plot(x_exact, y_exact, label='exact solution')

    print('Iterations')
    print(f'Step = {h}')
    print('shooting:', iters_shooting)
    print(f'Step = {h / 2}')
    print('shooting:', iters_shooting2)
    print()

    print('Mean absolute errors')
    print(f'Step = {h}')
    print('shooting:', mae(y_shooting, y_exact))
    print('finite difference:', mae(y_fd, y_exact))
    print(f'Step = {h / 2}')
    print('shooting:', mae(y_shooting2, y_exact2))
    print('finite difference:', mae(y_fd2, y_exact2))
    print()

    print('Runge-Romberg accuracy')
    print('shooting:', runge_romberg_method(h, h / 2, y_shooting, y_shooting2, 1))
    print('finite difference:', runge_romberg_method(h, h / 2, y_fd, y_fd2, 4))

    plt.legend()
    plt.show()
