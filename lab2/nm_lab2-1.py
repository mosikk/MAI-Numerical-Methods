"""
For input:
0 2

Output:
Iteration method
x = 0.4284960281514505 ; f(x) = -0.00036989405108200923
Iterations: 7
Newton method
x = 0.4282114790154919 ; f(x) = -1.0529842553452795e-08
Iterations: 3
"""

import math


def f(x):
    return math.log(x + 1) - 2*x + 0.5


def phi(x):
    return (math.log(x + 1) + 0.5) / 2


def df(x):
    return 1.0 / (x + 1) - 2


def iteration_method(f, phi, interval, eps):
    """
    Find root of f(x) == 0 at interval using iteration method
    Returns x and number of iterations
    """
    l, r = interval[0], interval[1]
    x_prev = (l + r) * 0.5
    iters = 0
    while True:
        iters += 1
        x = phi(x_prev)
        if abs(f(x) - f(x_prev)) < eps:
            break
        x_prev = x

    return x, iters


def newton_method(f, df, interval, eps):
    """
    Find root of f(x) == 0 at interval using newton method
    Returns x and number of iterations
    """
    l, r = interval[0], interval[1]
    x_prev = (l + r) * 0.5
    iters = 0
    while True:
        iters += 1
        x = x_prev - f(x_prev) / df(x_prev)
        if abs(f(x) - f(x_prev)) < eps:
            break
        x_prev = x

    return x, iters


if __name__ == "__main__":
    print('Enter interval coordinates')
    l, r = map(float, input().split())
    eps = float(input('Enter epsilon: '))

    print('Iteration method')
    x_iter, i_iter = iteration_method(f, phi, (l, r), eps)
    print('x =', x_iter, '; f(x) =', f(x_iter))
    print('Iterations:', i_iter)

    print('Newton method')
    x_newton, i_newton = newton_method(f, df, (l, r), eps)
    print('x =', x_newton, '; f(x) =', f(x_newton))
    print('Iterations:', i_newton)
