"""
Output:

Given cauchy problem:
y'' - 2tg(x)y' - 3y = 0
y(0) = 1
y'(0) = 3

We will convert this to system of equations with order = 1:
y' = g(x, y, z) = z
z' = f(x, y, z) = 2tg(x)y' + 3*y
y(0) = 1
z(0) = 3

Mean absolute errors
Step = 0.1
euler: 0.07812580875250418
runge-kutta: 0.00028203900579902954
adams: 0.02361929533428333
Step = 0.05
euler: 0.03480209609041373
runge-kutta: 1.749583670292088e-05
adams: 0.002400626888710371

Runge-Romberg accuracy
euler: 0.08183690964830621
runge-kutta: 9.5828260475453e-05
adams: 0.008086838571025467
"""

from math import sqrt, exp, cos
import math
import numpy as np
import matplotlib.pyplot as plt


def f(x, y, z):
    return 2 * math.tan(x) * z + 3 * y


def g(x, y, z):
    return z


def exact_solution(x):
    """
    Exact analytical solution of given cauchy problem (calculated by wolfram)
    Solution given in task was incorrect :(
    """
    return 0.25 * exp(-sqrt(2) * x) * ((2 + 3 * sqrt(2)) * exp(2 * sqrt(2) * x) + 2 - 3 * sqrt(2)) / cos(x)


def euler_method(f, g, y0, z0, interval, h):
    """
    Solve cauchy problem:
    y' = g(x, y, z)
    z' = f(x, y, z)
    y(0) = y0
    z(0) = z0
    where x in interval, h - step using euler method

    Returns solution - tabular function y=f(x) as two arrays: x, y
    """
    l, r = interval
    x = [i for i in np.arange(l, r + h, h)]
    y = [y0]
    z = z0
    for i in range(len(x) - 1):
        z += h * f(x[i], y[i], z)
        y.append(y[i] + h * g(x[i], y[i], z))
    return x, y


def runge_kutta_method(f, g, y0, z0, interval, h, return_z=False):
    """
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


def adams_method(f, g, y0, z0, interval, h):
    """
    Solve cauchy problem:
    y' = g(x, y, z)
    z' = f(x, y, z)
    y(0) = y0
    z(0) = z0
    where x in interval, h - step using adams method

    Returns solution - tabular function y=f(x) as two arrays: x, y
    """
    x_runge, y_runge, z_runge = runge_kutta_method(f, g, y0, z0, interval, h, return_z=True)
    x = x_runge
    y = y_runge[:4]
    z = z_runge[:4]
    for i in range(3, len(x_runge) - 1):
        z_i = z[i] + h * (55 * f(x[i], y[i], z[i]) -
                          59 * f(x[i - 1], y[i - 1], z[i - 1]) +
                          37 * f(x[i - 2], y[i - 2], z[i - 2]) -
                          9 * f(x[i - 3], y[i - 3], z[i - 3])) / 24
        z.append(z_i)
        y_i = y[i] + h * (55 * g(x[i], y[i], z[i]) -
                          59 * g(x[i - 1], y[i - 1], z[i - 1]) +
                          37 * g(x[i - 2], y[i - 2], z[i - 2]) -
                          9 * g(x[i - 3], y[i - 3], z[i - 3])) / 24
        y.append(y_i)
    return x, y


def runge_rombert_method(h1, h2, y1, y2, p):
    """
    Find more accuracy of solution using previous calculations.
    Works if h1 == 2 * h2
    """
    assert h1 == h2 * 2
    norm = 0
    for i in range(len(y1)):
        norm += (y1[i] - y2[i * 2]) ** 2
    return norm ** 0.5 / (2**p + 1)


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


def print_task():
    task = """
Given cauchy problem:
y'' - 2tg(x)y' - 3y = 0
y(0) = 1
y'(0) = 3

We will convert this to system of equations with order = 1:
y' = g(x, y, z) = z
z' = f(x, y, z) = 2tg(x)y' + 3*y
y(0) = 1
z(0) = 3
"""
    print(task)


if __name__ == '__main__':
    y0 = 1  # y(0) = 1
    dy0 = 3  # y'(0) = 3
    interval = (0, 1)  # x in [0; 1]
    h = 0.1
    print_task()

    x_euler, y_euler = euler_method(f, g, y0, dy0, interval, h)
    plt.plot(x_euler, y_euler, label=f'euler method, step={h}')
    x_euler2, y_euler2 = euler_method(f, g, y0, dy0, interval, h/2)
    plt.plot(x_euler2, y_euler2, label=f'euler method, step={h/2}')

    x_runge, y_runge = runge_kutta_method(f, g, y0, dy0, interval, h)
    plt.plot(x_runge, y_runge, label=f'runge kutta method, step={h}')
    x_runge2, y_runge2 = runge_kutta_method(f, g, y0, dy0, interval, h/2)
    plt.plot(x_runge2, y_runge2, label=f'runge kutta method, step={h/2}')

    x_adams, y_adams = adams_method(f, g, y0, dy0, interval, h)
    plt.plot(x_adams, y_adams, label=f'adams method, step={h}')
    x_adams2, y_adams2 = adams_method(f, g, y0, dy0, interval, h/2)
    plt.plot(x_adams2, y_adams2, label=f'adams method, step={h/2}')

    x_exact = [i for i in np.arange(interval[0], interval[1] + h, h)]
    x_exact2 = [i for i in np.arange(interval[0], interval[1] + h/2, h/2)]
    y_exact = [exact_solution(x_i) for x_i in x_exact]
    y_exact2 = [exact_solution(x_i) for x_i in x_exact2]
    plt.plot(x_exact, y_exact, label='exact solution')

    print('Mean absolute errors')
    print(f'Step = {h}')
    print('euler:', mae(y_euler, y_exact))
    print('runge-kutta:', mae(y_runge, y_exact))
    print('adams:', mae(y_adams, y_exact))
    print(f'Step = {h/2}')
    print('euler:', mae(y_euler2, y_exact2))
    print('runge-kutta:', mae(y_runge2, y_exact2))
    print('adams:', mae(y_adams2, y_exact2))
    print()

    print('Runge-Romberg accuracy')
    print('euler:', runge_rombert_method(h, h/2, y_euler, y_euler2, 1))
    print('runge-kutta:', runge_rombert_method(h, h/2, y_runge, y_runge2, 4))
    print('adams:', runge_rombert_method(h, h/2, y_adams, y_adams2, 4))

    plt.legend()
    plt.show()
