"""
This file contains functions designed to facilitate working with polynomials,
represented using lists of (possibly complex) coefficients. In particular, we
need to be able to form polynomials from roots and evaluate them on specific
inputs.
"""

def poly(roots):
    """
    Calculates the coefficients of a polynomial containing the given roots.
    For example, if roots = [3, -5, 1], poly(roots) = [-15, 2, 1], which can
    be interpreted as -15 * x^0 + 2 * x^1 + 1 * x^2.

    The method uses dynamic programming. Each cell in the table contains a
    partial result, and the last row of the table will contain the final
    coefficients.

    References:
      - https://cs.stackexchange.com/questions/98107/calculating-a-polynomials-coefficients-from-its-roots
    """
    # Easy case - no roots means a polynomial of 1
    if len(roots) == 0:
        return [1]

    N = len(roots)

    # Create the table with size [N + 1 x N]. Rows contain all coefficients for
    # the polynomial containing the first i coefficients. Columns contain
    # intermediate values needed to calculate subsequent elements in the table.
    table = [x[:] for x in [[1] * (N + 1)] * N]

    table[0][0] = -roots[0]
    for i in range(1, N):

        # Handle first column
        table[i][0] = -table[i - 1][0] * roots[i]

        # Each additional column depends on the information from the previous
        for k in range(1, i + 1):
            table[i][k] = -table[i - 1][k] * roots[i] + table[i - 1][k - 1]

    # The last row should contain our final coefficients
    return table[N - 1]

def evaluate(coefficients, point):
    """
    Evaluates the polynomial provided by the given coefficients in the order:
        c[0] * x^0 + c[1] * x^1 + c[2] * x^2 + ...
    using Horner's method. This method only requires N multiplications and N
    additions.

    References:
      - https://en.wikipedia.org/wiki/Horner%27s_method
    """
    res = coefficients[-1]
    for i in range(len(coefficients) - 1, 0, -1):
        res = coefficients[i - 1] + res * point
    return res
