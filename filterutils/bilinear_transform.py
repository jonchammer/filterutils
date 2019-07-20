import math

"""
The functions in this file are used to create difference equations based on the
coefficients of some analog filter using the Bilinear Transform.

Resources:
  - https://en.wikipedia.org/wiki/Bilinear_transform
  - https://www.robots.ox.ac.uk/~sjrob/Teaching/SP/l6.pdf
"""

def gen_K(freq, sampleRate, prewarped=False):
    """
    Create a suitable K based on the given frequency (in Hertz) and the sample
    rate (also in Hertz).

    When 'prewarped' is true, it is assumed that the provided frequency
    has already been adjusted based on the bilinear transform. If false, the
    frequency will be warped as part of the calculation of this function.
    """
    wc = 2 * math.pi * freq
    T  = 1 / sampleRate

    # Choose K based on whether or not prewarping has been applied
    return 2 / T if prewarped else wc / math.tan(wc * T / 2)

def general_difference_eq1(b, a, K):
    """
    Calculates the coefficients for a difference equation based on a general 1st
    order continuous-time filter of the form:

            b0 * s + b1   b0 + b1 * s^-1
    Ha(s) = ----------- = --------------
            a0 * s + a1   a0 + a1 * s^-1

    based on the Bilinear Transform.

    This function returns two lists of coefficients, b' and a'. They can be
    interpreted as:
    y[n] = b'[0] * x[n] + b'[1] * x[n - 1] - a'[0] * y[n - 1]
    """
    y0 = b[0] * K
    y1 = b[1] * 1

    z0 = a[0] * K
    z1 = a[1] * 1

    denom = z0 + z1

    c0 = ( y0 + y1) / denom
    c1 = (-y0 + y1) / denom

    c2 = (-z0 + z1) / denom

    return [c0, c1], [c2]

def general_difference_eq2(b, a, K):
    """
    Calculates the coefficients for a difference equation based on a general 2nd
    order continuous-time filter of the form:

            b0 * s^2 + b1 * s + b2   b0 + b1 * s^-1 + b2 * s^-2
    Ha(s) = ---------------------- = --------------------------
            a0 * s^2 + a1 * s + a2   a0 + a1 * s^-1 + a2 * s^-2

    based on the Bilinear Transform.

    The function returns two lists of coefficients, b' and a'. They can be
    interpreted as:
    y[n] = b'[0] * x[n] + b'[1] * x[n - 1] + b'[2] * x[n - 2]
                        - a'[0] * y[n - 1] - a'[1] * y[n - 2]
    """
    K2    = K * K

    y0 = b[0] * K2
    y1 = b[1] * K
    y2 = b[2] * 1

    z0 = a[0] * K2
    z1 = a[1] * K
    z2 = a[2] * 1

    denom = z0 + z1 + z2

    c0 = (     y0 + y1 +     y2) / denom
    c1 = (-2 * y0      + 2 * y2) / denom
    c2 = (     y0 - y1 +     y2) / denom

    c3 = (-2 * z0      + 2 * z2) / denom
    c4 = (     z0 - z1 +     z2) / denom
    return [c0, c1, c2], [c3, c4]

def general_difference_eq3(b, a, K):
    """
    Calculates the coefficients for a difference equation based on a general 3rd
    order continuous-time filter of the form:

            b0 * s^3 + b1 * s^2 + b2 * s + b3   b0 + b1 * s^-1 + b2 * s^-2 + b3 * s^-3
    Ha(s) = --------------------------------- = --------------------------------------
            a0 * s^3 + a1 * s^2 + a2 * s + a3   a0 + a1 * s^-1 + a2 * s^-2 + a3 * s^-3

    based on the Bilinear Transform.

    The function returns two lists of coefficients, b' and a'. They can be
    interpreted as:
    y[n] = b'[0] * x[n] + b'[1] * x[n - 1] + b'[2] * x[n - 2] + b'[3] * x[n - 3]
                        - a'[0] * y[n - 1] - a'[1] * y[n - 2] - a'[2] * y[n - 3]
    """
    K2    = K * K
    K3    = K * K * K

    y0    = b[0] * K3
    y1    = b[1] * K2
    y2    = b[2] * K
    y3    = b[3] * 1

    z0    = a[0] * K3
    z1    = a[1] * K2
    z2    = a[2] * K
    z3    = a[3] * 1

    denom = z0 + z1 + z2 + z3

    c0 = (     y0 + y1 + y2     + y3) / denom
    c1 = (-3 * y0 - y1 + y2 + 3 * y3) / denom
    c2 = ( 3 * y0 - y1 - y2 + 3 * y3) / denom
    c3 = (    -y0 + y1 - y2     + y3) / denom

    c4 = (-3 * z0 - z1 + z2 + 3 * z3) / denom
    c5 = ( 3 * z0 - z1 - z2 + 3 * z3) / denom
    c6 = (    -z0 + z1 - z2 +     z3) / denom
    return [c0, c1, c2, c3], [c4, c5, c6]

def general_difference_eq4(b, a, K):
    """
    Calculates the coefficients for a difference equation based on a general 4th
    order continuous-time filter of the form:

            b0 * s^4 + b1 * s^3 + b2 * s^2 + b3 * s + b4   b0 + b1 * s^-1 + b2 * s^-2 + b3 * s^-3 + b4 * s^-4
    Ha(s) = -------------------------------------------- = --------------------------------------------------
            a0 * s^4 + a1 * s^3 + a2 * s^2 + a3 * s + a4   a0 + a1 * s^-1 + a2 * s^-2 + a3 * s^-3 + a4 * s^-4

    based on the Bilinear Transform.

    The function returns two lists of coefficients, b' and a'. They can be
    interpreted as:
    y[n] = b'[0] * x[n] + b'[1] * x[n - 1] + b'[2] * x[n - 2] + b'[3] * x[n - 3] + b'[4] * x[n - 4]
                        - a'[0] * y[n - 1] - a'[1] * y[n - 2] - a'[2] * y[n - 3] - a'[3] * y[n - 4]
    """
    K2    = K * K
    K3    = K * K * K
    K4    = K * K * K * K

    y0    = b[0] * K4
    y1    = b[1] * K3
    y2    = b[2] * K2
    y3    = b[3] * K
    y4    = b[4] * 1

    z0    = a[0] * K4
    z1    = a[1] * K3
    z2    = a[2] * K2
    z3    = a[3] * K
    z4    = a[4] * 1

    denom = z0 + z1 + z2 + z3 + z4

    c0 = (     y0 +     y1 +     y2 +     y3 +     y4) / denom
    c1 = (-4 * y0 - 2 * y1          + 2 * y3 + 4 * y4) / denom
    c2 = ( 6 * y0          - 2 * y2          + 6 * y4) / denom
    c3 = (-4 * y0 + 2 * y1          - 2 * y3 + 4 * y4) / denom
    c4 = (     y0 -     y1 +     y2 -     y3 +     y4) / denom

    c5 = (-4 * z0 - 2 * z1          + 2 * z3 + 4 * z4) / denom
    c6 = ( 6 * z0          - 2 * z2          + 6 * z4) / denom
    c7 = (-4 * z0 + 2 * z1          - 2 * z3 + 4 * z4) / denom
    c8 = (     z0 -     z1 +     z2 -     z3 +     z4) / denom
    return [c0, c1, c2, c3, c4], [c5, c6, c7, c8]
