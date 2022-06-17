"""
For input:

14 9
-8 14 6
-5 -17 8
1 5 -2
-4 -10

125 -56 144 36 70

Output:
Solution
[7.0, 3.0, -6.999999999999999, 5.000000000000001, -9.0]

"""


def read_tridiagonal_matrix(n):
    """
    Get tridiagonal matrix with n rows by reading only not-null elements
    """
    A = [[0 for _ in range(n)] for _ in range(n)]
    A[0][0], A[0][1] = map(int, input().split())
    for i in range(1, n-1):
        A[i][i-1], A[i][i], A[i][i+1] = map(int, input().split())
    A[n-1][n-2], A[n-1][n-1] = map(int, input().split())
    return A


def tridiagonal_solve(A, b):
    """
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


if __name__ == "__main__":
    n = int(input('Enter the number of equations: '))
    print('Enter not null elements of tridiagonal matrix')
    A = read_tridiagonal_matrix(n)
    print('Enter vector b')
    b = list(map(int, input().split()))

    print('Solution')
    x = tridiagonal_solve(A, b)
    print(x)
