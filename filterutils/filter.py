from . import polynomials
import cmath
import numpy as np

"""
This file contains classes and functions used to represent, create, and
manipulate general-purpose digital IIR (or FIR) filters.

References:
  - http://www.earlevel.com/main/2016/12/08/filter-frequency-response-grapher/
"""

class Filter():
    """
    This class represents a basic IIR (or FIR) filter, based on a transfer
    function H(z) of the form:

           b[0] * z^0 + b[1] * z^-1 + b[2] * z^-2 + ...
    H(z) = ---------------------------------------------
              1 * z^0 + a[0] * z^-1 + a[1] * z^-2 + ...

    The numerator of the filter represents the nonrecursive components (the FIR)
    part, and the denominator represents the recursive components. When no a[i]
    are provided, the denominator will be 1, leaving a simple FIR filter.

    This transfer function can be directly interpreted as a difference equation
    of the form:

    y[n] = b[0] * x[n] + b[1] * x[n - 1] + b[2] * x[n - 2] + b[3] * x[n - 3] + ...
                       - a[0] * y[n - 1] - a[1] * y[n - 2] - a[2] * y[n - 3] - ...

    which is used when using the filter. Note that the a[i] terms are additive
    in the transfer function, but are subtracted when used in the difference
    equation.
    """

    def __init__(self, b=None, a=None, coefficients=None):
        """
        Creates a filter given array-like objects b and a. B represents the
        non-recursive filter coefficients (numerator of transfer function), and
        A (if provided) represents the recursive coefficients (denominator of
        the transfer function). A list containing both sets can also be used
        (coefficients).

        B should be of the form:
        b[0] * z^0 + b[1] * z^-1 + b[2] * z^-2 + ...

        A should be normalized so the first coefficient is 1, resulting in:
        1 * z^0 + a[0] * z^-1 + a[1] * z^-2 + ...

        As a result, len(a) == len(b) - 1 when both the numerator and denominator
        have the same number of coefficients.
        """
        if b is not None:
            self.b = b
            self.a = a or []
        elif coefficients is not None:
            self.b = coefficients[0]
            self.a = coefficients[1] or []
        else:
            assert false, "Either 'b' or 'coefficients' must be provided."

    def apply(self, x, x_hist=None, y_hist=None):
        """
        Applies this filter to the provided input signal, 'x'. This function
        returns filtered result, 'y', which is a list of the same size as 'x'.

        The arguments 'x_hist' and 'y_hist' can be used when filtering data in
        batches. 'x_hist' should contain the len(b) - 1 previous input examples,
        and 'y_hist' should contain the len(a) previous outputs. If not provided,
        it is assumed that all previous inputs and outputs are 0.

        This function returns a tuple (y, x_hist, y_hist) consisting of the
        filtered signal, the next values of 'x_hist', and the next values of
        'y_hist'. Typical usage would look like this:

        x = [...] # Get buffer from somewhere
        y, x_hist, y_hist = filter.apply(x)
        while True:
            x = [...] # Get next buffer
            y, x_hist, y_hist = filter.apply(x, x_hist, y_hist)

        In the event the entire signal is available, the 'x_hist' and 'y_hist'
        outputs can be ignored. E.g.:

        x = [...] # Get buffer from somewhere
        y, _, _ = filter.apply(x)

        NOTE: This implementation avoids maintaining filter state inside the
        object itself in order to easier facilitate multithreading or re-entry.
        It would, of course, be fairly trivial to wrap that information in a
        'filter context' object that holds state between iterations, if desired.
        """
        N      = len(self.b)
        M      = len(self.a)
        P      = len(x)
        result = [0] * P

        # Prepare the history arrays. We never modify the original value if it
        # was provided
        if x_hist is None:
            x_hist = [0] * (N - 1)
        else:
            x_hist = [0] * (N - 1 - len(x_hist)) + x_hist

        if y_hist is None:
            y_hist = [0] * M
        else:
            y_hist = [0] * (M - len(y_hist)) + y_hist

        # General formula:
        # y[n] = b[0] * x[n] + b[1] * x[n - 1] + b[2] * x[n - 2] + b[3] * x[n - 3] + ...
        #                    - a[0] * y[n - 1] - a[1] * y[n - 2] - a[2] * y[n - 3] - ...
        for i in range(P):

            # Handle nonrecursive elements. We take advantage of Python's
            # negative index slicing for the x_hist and y_hist elements. In
            # other languages, we would write 'x_hist[N - i - j]' to access the
            # elements in reverse order.
            for j in range(N):
                if i - j >= 0:
                    result[i] += self.b[j] * x[i - j]
                else:
                    result[i] += self.b[j] * x_hist[i - j]

            # Handle recursive elements
            for j in range(M):
                if i - j - 1 >= 0:
                    result[i] -= self.a[j] * result[i - j - 1]
                else:
                    result[i] -= self.a[j] * y_hist[i - j - 1]

        return result, x[-N-1:], result[-M:]

    def impulse_response(self, steps):
        """
        Calculates the impulse response of this filter to the given number of
        steps. This is equivalent to calling apply() on the signal
        [1, 0, 0, ... 0], but a slightly more efficient implementation is used.
        """
        assert steps >= len(self.b)

        # General formula:
        # y[n] = b[0] * x[n] + b[1] * x[n - 1] + b[2] * x[n - 2] + b[3] * x[n - 3] + ...
        #                    - a[0] * y[n - 1] - a[1] * y[n - 2] - a[2] * y[n - 3] - ...
        result = [0] * steps

        # The first len(b) elements involve both the non-recursive components and
        # the recursive ones
        N = len(self.b)
        M = len(self.a)
        for i in range(N):
            result[i] = self.b[i]
            for j in range(min(i, M)):
                result[i] -= self.a[j] * result[i - 1 - j]

        # Recursive components only from here until we hit the step limit
        for i in range(N, steps):
            for j in range(M):
                result[i] -= self.a[j] * result[i - 1 - j]

        return result

    def frequency_response(self, steps=512, fft=False):
        """
        Returns the frequency response of this filter calculated using one of
        two different methods:

        1) Evaluate the transfer function H(z) = B/A directly for 'steps' points
           evenly spaced on the lower half of the unit circle, z = e^(-i * w).
        2) Calculate the FFT of the impulse response.

        Which implementation provides best results will likely vary based on the
        filter characteristics (e.g. number of coefficients, cutoff frequency,
        etc.).

        The result will be a np.complex128 numpy array with 'steps' elements.
        To switch to polar coordinates (amplitude and phase), the np.abs()
        and np.angle() functions can be used.

        NOTE: Especially when using the FFT implementation, a large number of
        steps may be required for some pathological situations, e.g. a
        Butterworth lowpass filter with a very low cutoff frequency. The defualt
        value for 'steps' should work well enough in most cases, though.
        """

        # Default approach: evaluate the frequency response directly by
        # evaluating the transfer function at 'steps' frequencies on the lower
        # half of the unit circle in the z-plane.
        if not fft:
            vals = np.zeros(steps, dtype=np.complex128)
            for i in range(steps):
                z = cmath.exp(-1j * cmath.pi * i / steps)
                vals[i] = self.eval_transfer_function(z)
            return vals

        # Alternate approach: calculate the FFT of the impulse response.
        else:
            ir = self.impulse_response(2 * steps)
            return np.fft.fft(ir, 2 * steps)[0 : steps]

    def eval_transfer_function(self, z):
        """
        The transfer function for this filter is given by H(z) = B/A, where B
        and A are the filter coefficients. This function evaluates H(z) for
        any arbitrary complex number, z.
        """
        return polynomials.evaluate(self.b, z) / polynomials.evaluate([1] + self.a, z)

# ---------------------------------------------------------------------------- #

class CascadedFilter():
    """
    Cascaded filters are wrappers around several filters applied in series on
    an input signal. This allows for implementations of the digital biquad
    technique, for example. Cascaded filters can also be used to implement
    wideband bandpass or bandstop filters (by adding both a lowpass filter and a
    highpass filter).
    """

    def __init__(self, filters):
        """
        Create a CascadedFilter using the provided collection of filters.
        Filters will be applied in the order in which they were provided.
        """
        self.filters = filters

    def apply(self, x, x_hist=None, y_hist=None):
        """
        Applies the cascade filter on the provided signal. The signal is routed
        through each filter in series, and the output of the final filter is
        returned to the user.

        See Filter.apply() for descriptions of the 'x_hist' and 'y_hist'
        arguments, as well as the return value.
        """

        if x_hist is None:
            x_hist = [ None ] * len(self.filters)
        else:
            x_hist = x_hist[:]

        if y_hist is None:
            y_hist = [ None ] * len(self.filters)
        else:
            y_hist = y_hist[:]

        input  = x
        output = None
        for i, filter in enumerate(self.filters):
            output, x_hist[i], y_hist[i] = filter.apply(input, x_hist[i], y_hist[i])
            input = output
        return output, x_hist, y_hist

    def impulse_response(self, steps=512):
        """
        Calculates the impulse response of this filter to the given number of
        steps. This is equivalent to calling apply() on each provided filter
        using a signal of [1, 0, 0, ... 0]
        """
        input = self.filters[0].impulse_response(steps)
        if len(self.filters) == 1:
            return input

        output = None
        for i in range(1, len(self.filters)):
            output, _, _ = self.filters[i].apply(input)
            input  = output
        return output

    def frequency_response(self, steps=512, fft=False):
        """
        Returns the frequency response of this filter calculated using one of
        two different methods:

        1) Evaluate the transfer function H(z) = B/A directly for 'steps' points
           evenly spaced on the lower half of the unit circle, z = e^(-i * w).
        2) Calculate the FFT of the impulse response.

        Which implementation provides best results will likely vary based on the
        filter characteristics (e.g. number of coefficients, cutoff frequency,
        etc.).

        The result will be a np.complex128 numpy array with 'steps' elements.
        To switch to polar coordinates (amplitude and phase), the np.abs()
        and np.angle() functions can be used.

        NOTE: Especially when using the FFT implementation, a large number of
        steps may be required for some pathological situations, e.g. a
        Butterworth lowpass filter with a very low cutoff frequency. The defualt
        value for 'steps' should work well enough in most cases, though.
        """

        # Default approach: evaluate the frequency response directly by
        # evaluating the transfer function at 'steps' frequencies on the lower
        # half of the unit circle in the z-plane.
        if not fft:
            vals = np.zeros(steps, dtype=np.complex128)
            for i in range(steps):
                z = cmath.exp(-1j * cmath.pi * i / steps)
                vals[i] = self.eval_transfer_function(z)
            return vals

        # Alternate approach: calculate the FFT of the impulse response.
        else:
            ir = self.impulse_response(2 * steps)
            return np.fft.fft(ir, 2 * steps)[0 : steps]

    def eval_transfer_function(self, z):
        """
        The transfer function for this cascade is given by:
          H(z) = (B1/A1) * (B2/A2) * (B3/A3) * ...
        where B and A are the filter coefficients for each individual filter
        in the cascade. This function evaluates H(z) for any arbitrary complex
        number, z.
        """
        product = 1
        for f in self.filters:
            product *= f.eval_transfer_function(z)
        return product

# ---------------------------------------------------------------------------- #

def group_pairs(values):
    """
    Given an array of (complex) numbers, this function groups them into:
      1) pairs of complex conjugates
      2) pairs of +/- real numbers
      3) singletons. E.g.:

    complex_pairs([1.0+2j, 5.0-7j, 8.0+0j, 1.0-2j, -3+1j, -8.0+0j])
      -> [ 1.0+2j, 1.0-2j, 8.0+0j, -8+0j, 5.0-7j, -3+1j ]

    Complex conjugates will appear first in the list, followed by real pairs,
    followed by singletons.
    """

    # Returns true when the two provided numbers are complex conjugates
    def conjugate(n1, n2, threshold):
        return abs(n1.real - n2.real) < threshold and \
               abs(n1.imag + n2.imag) < threshold and \
               n1.imag != n2.imag

    # Returns true when the provided number is (effectively) real
    def is_real(n, threshold):
        return abs(n.imag) < threshold

    # Returns true when the two provided numbers are real pairs (+- some value)
    def real_pair(n1, n2, threshold):
        return is_real(n1, threshold) and is_real(n2, threshold) and \
               abs(-n1.real - n2.real) < threshold

    # Returns the first element in the provided collection that matches pred.
    def first_true(iterable, default=None, pred=None):
        return next(filter(pred, iterable), default)

    threshold     = 1E-14
    cp            = values[:]
    complex_pairs = []
    real_pairs    = []
    singletons    = []

    # For each element in the cloned list, try to find its match. If we find it,
    # both the element and the match are removed from consideration. If not,
    # just the single element is removed.
    while len(cp) > 0:
        v1 = cp[0]

        # Try complex conjugates
        v2 = first_true(cp[1:], pred=lambda x: conjugate(v1, x, threshold))
        if v2 is not None:
            complex_pairs.extend([v1, v2])
            cp.remove(v1)
            cp.remove(v2)
            continue

        # Try real pairs
        v2 = first_true(cp[1:], pred=lambda x: real_pair(v1, x, threshold))
        if v2 is not None:
            real_pairs.extend([v1, v2])
            cp.remove(v1)
            cp.remove(v2)
            continue

        # v1 must be a singleton
        singletons.append(v1)
        cp.remove(v1)

    return complex_pairs + real_pairs + singletons

def create_digital_filter(poles, zeroes, z0=None, cascade=False):
    """
    Returns either a single digital filter or a cascade of biquad filters based
    on the provided poles and zeroes. The zeroes are used to create the
    numerator of the transfer function H(z), and the poles are used to create
    the denominator.

    When z0 is provided, the overall gain of the filter is adjusted such that
    the gain at z0 is equal to 1.0. When z0 is not provided, the filter will
    not be scaled.

    When 'cascade' is True, a cascaded biquad will be returned instead of a
    single filter. The cascaded biquad will likely be more numerically stable
    than the direct implementation.

    NOTE: Poles and zeroes may be complex numbers, and the returned filter may
    have complex coefficients, but an effort is made to use real coefficients
    when possible.
    """

    def single(poles, zeroes, z0):
        a = polynomials.poly(poles)
        b = polynomials.poly(zeroes)

        # Try to use real coefficients when possible
        threshold = 1E-14
        a = [ i.real if abs(i.imag) < threshold else i for i in a ]
        b = [ i.real if abs(i.imag) < threshold else i for i in b ]

        # The polynomials are written as [x^0, x^1, x^2, etc]. We need the reverse
        # order to calculate [x^0, x^-1, x^-2, etc.] for evaluation of the transfer
        # function. The opposite order will also be needed for the filter itself.
        b.reverse()
        a.reverse()

        if z0 is not None:
            # Scale coefficients so that the amplitude at z0 is 1.0. The inverse
            # of the magnitude of the output gives us our K that we can use to
            # scale the numerator of the transfer function (b) accordingly.
            K  = abs(polynomials.evaluate(a, z0) / polynomials.evaluate(b, z0))
            b  = [ K * i for i in b ]

        # a[0] should always be 1, so it's ignored for the filter.
        assert a[0] - 1 < 1E-5, "A polynomial is not normalized."
        return Filter(b=b, a=a[1:])

    # Convert poles and zeroes to complex numbers if they're not already
    poles  = [ p if isinstance(p, complex) else complex(p) for p in poles  ]
    zeroes = [ z if isinstance(z, complex) else complex(z) for z in zeroes ]

    # Use a direct implementation
    if not cascade:
        return single(poles, zeroes, z0)

    # Use a cascaded implementation
    else:
        p_pairs = group_pairs(poles)
        z_pairs = group_pairs(zeroes)

        # Group the poles and zeroes into groups of 2 to form each individual
        # biquad. When one or the other runs out, the 'biquad_poles' or
        # 'biquad_zeroes' lists will be empty, resulting in a polynomial of 1.
        p_index = 0
        z_index = 0
        filters = []
        while p_index < len(p_pairs) or z_index < len(z_pairs):
            biquad_poles  = p_pairs[p_index: p_index + 2]
            biquad_zeroes = z_pairs[z_index: z_index + 2]
            filters.append(single(biquad_poles, biquad_zeroes, z0))

            p_index += 2
            z_index += 2
        return CascadedFilter(filters)
