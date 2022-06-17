"""
Output:
Lagrange interpolation
Points A
Polynom
L(x) = + -1.16*(x-0.52)(x-1.05)(x-1.57) + 4.84*(x-0.00)(x-1.05)(x-1.57) + -5.39*(x-0.00)(x-0.52)(x-1.57) + 1.82*(x-0.00)(x-0.52)(x-1.05)
Abs error for test point = 0.00035306942058976887
Points B
Polynom
L(x) = + -1.55*(x-0.52)(x-0.79)(x-1.57) + 9.68*(x-0.00)(x-0.79)(x-1.57) + -9.24*(x-0.00)(x-0.52)(x-1.57) + 1.22*(x-0.00)(x-0.52)(x-0.79)
Abs error for test point = 0.0016970143507963886

Newton interpolation
Points A
Polynom
P(x) = 1.00 + (x-0.00)*0.74 + (x-0.00)(x-0.52)*-0.42 + (x-0.00)(x-0.52)(x-1.05)*0.11
Abs error for test point = 0.00035306942058976887
Points B
Polynom
P(x) = 1.00 + (x-0.00)*0.74 + (x-0.00)(x-0.52)*-0.45 + (x-0.00)(x-0.52)(x-0.79)*0.11
Abs error for test point = 0.0016970143507961666
"""

import math


def f(x):
    return math.cos(x) + x


def lagrange_interpolation(x, y, test_point):
    """
    x - array of coords by x
    y - array of coords by y
    test_point = (x*, y*) - point for checking error of interpolation
    Returns:
        lagrange interpolation function for points x, y as string,
        error between test_point and interpolation
    """
    assert len(x) == len(y)
    polynom_str = 'L(x) ='
    polynom_test_value = 0  # L(x*)
    for i in range(len(x)):
        cur_enum_str = ''  # enumerator of polynom's current term as string
        cur_enum_test = 1  # current value of polynom for test point
        cur_denom = 1
        for j in range(len(x)):
            if i == j:
                continue
            cur_enum_str += f'(x-{x[j]:.2f})'
            cur_enum_test *= (test_point[0] - x[j])
            cur_denom *= (x[i] - x[j])

        polynom_str += f' + {(y[i] / cur_denom):.2f}*' + cur_enum_str
        polynom_test_value += y[i] * cur_enum_test / cur_denom
    return polynom_str, abs(polynom_test_value - test_point[1])


def newton_interpolation(x, y, test_point):
    """
    x - array of coords by x
    y - array of coords by y
    test_point = (x*, y*) - point for checking error of interpolation
    Returns:
        newton interpolation function for points x, y as string,
        error between test_point and interpolation
    """
    assert len(x) == len(y)

    # Find coefficients for polynom
    n = len(x)
    coefs = [y[i] for i in range(n)]
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            coefs[j] = float(coefs[j] - coefs[j - 1]) / float(x[j] - x[j - i])

    # Get polynom
    polynom_str = 'P(x) = '
    polynom_test_value = 0  # P(x*)

    cur_multipliers_str = ''
    cur_multipliers = 1
    for i in range(n):
        polynom_test_value += cur_multipliers * coefs[i]
        if i == 0:
            polynom_str += f'{coefs[i]:.2f}'
        else:
            polynom_str += ' + ' + cur_multipliers_str + '*' + f'{coefs[i]:.2f}'

        cur_multipliers *= (test_point[0] - x[i])
        cur_multipliers_str += f'(x-{x[i]:.2f})'
    return polynom_str, abs(polynom_test_value - test_point[1])


if __name__ == '__main__':
    x_a = [0, math.pi / 6, 2 * math.pi / 6, 3 * math.pi / 6]
    x_b = [0, math.pi / 6, math.pi / 4, math.pi / 2]
    y_a = [f(x) for x in x_a]
    y_b = [f(x) for x in x_b]

    x_test = 1
    y_test = f(x_test)

    print('Lagrange interpolation')
    print('Points A')
    lagrange_polynom_a, lagrange_error_a = lagrange_interpolation(x_a, y_a, (x_test, y_test))
    print('Polynom')
    print(lagrange_polynom_a)
    print('Abs error for test point =', lagrange_error_a)

    print('Points B')
    lagrange_polynom_b, lagrange_error_b = lagrange_interpolation(x_b, y_b, (x_test, y_test))
    print('Polynom')
    print(lagrange_polynom_b)
    print('Abs error for test point =', lagrange_error_b)
    print()

    print('Newton interpolation')
    print('Points A')
    newton_polynom_a, newton_error_a = newton_interpolation(x_a, y_a, (x_test, y_test))
    print('Polynom')
    print(newton_polynom_a)
    print('Abs error for test point =', newton_error_a)

    print('Points B')
    newton_polynom_b, newton_error_b = newton_interpolation(x_b, y_b, (x_test, y_test))
    print('Polynom')
    print(newton_polynom_b)
    print('Abs error for test point =', newton_error_b)
