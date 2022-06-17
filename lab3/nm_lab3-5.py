"""
Output:
Rectangle method
Step = 1.0: integral = -0.1878310847553158
Step = 0.5: integral = -0.19940628065257357
Trapeze method
Step = 1.0: integral = -0.23658183921341816
Step = 0.5: integral = -0.212206461984367
Simpson method
Step = 1.0: integral = -0.2071717755928282
Step = 0.5: integral = -0.20408133624134991
Runge Rombert method
More accurate integral by rectangle method = -0.20105988006646755
More accurate integral by trapeze method = -0.20872426523735968
More accurate integral by Simpson method = -0.20363984490542444
"""


def f(x):
    return x**2 / (x**3 - 27)


def integrate_rectangle_method(f, l, r, h):
    """
    Calculate integral f(x)dx at interval [l; r] using rectangle method with step=h
    """
    result = 0
    cur_x = l
    while cur_x < r:
        result += h * f((cur_x + cur_x + h) * 0.5)
        cur_x += h
    return result


def integrate_trapeze_method(f, l, r, h):
    """
    Calculate integral f(x)dx at interval [l; r] using trapeze method with step=h
    """
    result = 0
    cur_x = l
    while cur_x < r:
        result += h * 0.5 * (f(cur_x + h) + f(cur_x))
        cur_x += h
    return result


def integrate_simpson_method(f, l, r, h):
    """
    Calculate integral f(x)dx at interval [l; r] using simpson method with step=h
    """
    result = 0
    cur_x = l + h
    while cur_x < r:
        result += f(cur_x - h) + 4*f(cur_x) + f(cur_x + h)
        cur_x += 2 * h
    return result * h / 3


def runge_rombert_method(h1, h2, integral1, integral2, p):
    """
    Find more accurate value of integral using previous calculations.
    Works if h1 == k * h2
    """
    return integral1 + (integral1 - integral2) / ((h2 / h1)**p - 1)


if __name__ == '__main__':
    l, r = -2, 2  # interval of integrating
    h1, h2 = 1.0, 0.5  # steps

    print('Rectangle method')
    int_rectangle_h1 = integrate_rectangle_method(f, l, r, h1)
    int_rectangle_h2 = integrate_rectangle_method(f, l, r, h2)
    print(f'Step = {h1}: integral = {int_rectangle_h1}')
    print(f'Step = {h2}: integral = {int_rectangle_h2}')

    print('Trapeze method')
    int_trapeze_h1 = integrate_trapeze_method(f, l, r, h1)
    int_trapeze_h2 = integrate_trapeze_method(f, l, r, h2)
    print(f'Step = {h1}: integral = {int_trapeze_h1}')
    print(f'Step = {h2}: integral = {int_trapeze_h2}')

    print('Simpson method')
    int_simpson_h1 = integrate_simpson_method(f, l, r, h1)
    int_simpson_h2 = integrate_simpson_method(f, l, r, h2)
    print(f'Step = {h1}: integral = {int_simpson_h1}')
    print(f'Step = {h2}: integral = {int_simpson_h2}')

    print('Runge Rombert method')
    print(f'More accurate integral by rectangle method = {runge_rombert_method(h1, h2, int_rectangle_h1, int_rectangle_h2, 3)}')
    print(f'More accurate integral by trapeze method = {runge_rombert_method(h1, h2, int_trapeze_h1, int_trapeze_h2, 3)}')
    print(f'More accurate integral by Simpson method = {runge_rombert_method(h1, h2, int_simpson_h1, int_simpson_h2, 3)}')
