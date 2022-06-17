"""
For input:
0.4 0.45
0.95 1
0.0000001

Output:
Newton method
x = [0.42490233 0.97717168] ; f1(x) = 1.9590329358720737e-11 ; f2(x)= 5.220823773299799e-13
Iterations: 3
Iteration method
x = [0.42040958033662124, 0.9771610665596537] ; f1(x) = -0.0009701961902964795 ; f2(x) = 0.011327503961215957
Iterations: 1
"""


import math
import numpy as np


def f1(X):
    return X[0]**2 / 4 + X[1]**2 - 1


def f2(X):
    return 2*X[1] - math.exp(X[0]) - X[0]


def df1_dx1(X):
    # df1 / dx1
    return X[0] / 2


def df1_dx2(X):
    return 2 * X[1]


def df2_dx1(X):
    return -math.exp(X[0]) - 1


def df2_dx2(X):
    return 2


def phi1(X):
    # X1 = phi1(X)
    return 2 * X[1] - math.exp(X[0])


def phi2(X):
    # X2 = phi2(X)
    return math.sqrt(1 - X[0] ** 2 / 4)


def dphi1_dx1(X):
    return -math.exp(X[0])


def dphi1_dx2(X):
    return 2


def dphi2_dx1(X):
    return 1 / (2 * math.sqrt(1 - X[0] ** 2 / 4)) * (-2 * X[0] / 4)


def dphi2_dx2(X):
    return 0


def L_inf_norm(a):
    """
    Get L_inf norm of array a
    ||a||_inf = max(abs(a))
    """
    abs_a = [abs(i) for i in a]
    return max(abs_a)


def get_q(interval1, interval2):
    """
    Get q coefficient for iteration method
    """
    l1, r1 = interval1
    l2, r2 = interval2
    m1 = (l1 + r1) / 2
    m2 = (l2 + r2) / 2
    x1 = m1 + abs(r1 - l1)
    x2 = m2 + abs(r2 - l2)
    max1 = abs(dphi1_dx1([x1, x2])) + abs(dphi1_dx2([x1, x2]))
    max2 = abs(dphi2_dx1([x1, x2])) + abs(dphi2_dx2([x1, x2]))
    return max(max1, max2)


def iteration_method(phi1, phi2, intervals, eps):
    """
    Find root of system:
    f1(x1, x2) == 0
    f2(x1, x2) == 0
    at interval using iteration method

    x1 = phi1(x1, x2)
    x2 = phi2(x1, x2)

    Returns (x1, x2) and number of iterations
    """
    l1, r1 = intervals[0][0], intervals[0][1]
    l2, r2 = intervals[1][0], intervals[1][1]
    x_prev = [(l1 + r1) * 0.5, (l2 + r2) * 0.5]
    q = get_q(intervals[0], intervals[1])
    iters = 0
    while True:
        iters += 1
        x = [phi1(x_prev), phi2(x_prev)]
        if q / (1 - q) * L_inf_norm([(x[i] - x_prev[i]) for i in range(len(x))]) < eps:
            break
        x_prev = x

    return x, iters


def newton_method(f1, f2, df1_dx1, df1_dx2, df2_dx1, df2_dx2, intervals, eps):
    """
    Find root of system:
    f1(x1, x2) == 0
    f2(x1, x2) == 0
    at intervals using newton method
    Returns x1, x2 and number of iterations
    """
    l1, r1 = intervals[0][0], intervals[0][1]
    l2, r2 = intervals[1][0], intervals[1][1]
    x_prev = np.array([(l1 + r1) / 2, (l2 + r2) / 2])
    jacobi = []
    jacobi.append([df1_dx1(x_prev), df1_dx2(x_prev)])
    jacobi.append([df2_dx1(x_prev), df2_dx2(x_prev)])
    jacobi_inversed = np.linalg.inv(np.array(jacobi))
    iters = 0
    while True:
        iters += 1
        x = x_prev - jacobi_inversed @ np.array([f1(x_prev), f2(x_prev)])
        if L_inf_norm([(x[i] - x_prev[i]) for i in range(len(x))]) < eps:
            break
        x_prev = x

    return x, iters


if __name__ == "__main__":
    print('Enter interval for x1')
    l1, r1 = map(float, input().split())
    print('Enter interval for x2')
    l2, r2 = map(float, input().split())
    eps = float(input('Enter epsilon: '))

    print('Newton method')
    x_newton, i_newton = newton_method(f1, f2, df1_dx1, df1_dx2, df2_dx1, df2_dx2, [(l1, r1), (l2, r2)], eps)
    print('x =', x_newton, '; f1(x) =', f1(x_newton), '; f2(x)=', f2(x_newton))
    print('Iterations:', i_newton)

    print('Iteration method')
    x_iter, i_iter = iteration_method(phi1, phi2, [(l1, r1), (l2, r2)], eps)
    print('x =', x_iter, '; f1(x) =', f1(x_iter), '; f2(x) =', f2(x_iter))
    print('Iterations:', i_iter)
