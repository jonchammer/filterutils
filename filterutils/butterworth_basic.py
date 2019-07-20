import math
from . import bilinear_transform as bt
from . import filter

"""
The functions in this file allow for creation of Butterworth filters of various
types (e.g. lowpass, highpass, bandpass, notch) suitable for any DSP application,
such as audio filtering.

The normal construction is as follows:
  1) Choose an n-pole base formulation.
  2) Substitute the appropriate formula for a to transform into lowpass,
     highpass, etc. and simplify to polynomials in s for both the numerator and
     denominator.
  3) Use the bilinear transform to derive an appropriate IIR difference equation.

NOTES:
  1) Applying the bandpass or notch transformations to a 1-pole formulation will
     actually result in a 2-pole bandpass / notch filter. Therefore, all
     bandpass / notch filters will have an even number of poles.
  2) For filters with 2 or more poles, it is possible to specify a Q value that
     modifies the gain at wc. This is commonly used in audio synthesis for
     resonance. It can also be used to derive cascade biquad filter
     implementations.
  3) The Q provided for bandpass / notch filters is unfortunately not the same
     as the Q mentioned in the previous note. The bandpass / notch Q is related
     to the bandwidth, rather than the gain at wc.

--------------------------------------------------------------------------------
General Butterworth Formulations
--------------------------------------------------------------------------------
1-pole:                           3-pole:
          1                                       1
H(a) = -------                    H(a) = --------------------
        a + 1                            (a + 1)(a^2 + a + 1)

2-pole:                           4-pole:
                 1                                        1
H(a) = ---------------------      H(a) = ------------------------------------
       a^2 + sqrt(2) * a + 1             (a^2 + c1 * s + 1)(a^2 + c2 * s + 1)
                                    c1 = -2 * cos(5 * pi / 8)
                                    c2 = -2 * cos(7 * pi / 8)

--------------------------------------------------------------------------------
General Transformations
--------------------------------------------------------------------------------
                s                         Q(s^2 + wc^2)     s^2 + wc^2
 Lowpass: a = ----         Bandpass: a = -------------- = --------------
               wc                            wc * s         (wc/Q) * s

               wc                           wc * s          (wc/Q) * s
Highpass: a = ----            Notch: a = -------------- = --------------
                s                        Q(s^2 + wc^2)      s^2 + wc^2

--------------------------------------------------------------------------------
General References
--------------------------------------------------------------------------------
  - https://en.wikipedia.org/wiki/Butterworth_filter
  - https://en.wikipedia.org/wiki/Prototype_filter
"""

def butterworth_lp1(freq, sample_rate, prewarped=False):
    """
    Constructs a 1st order Butterworth lowpass filter with the provided cutoff
    frequency (in Hz) based on the provided sample rate (in Hz) using the
    following transfer equation:

             wc        0 * s^0 + wc * s^-1
    H(s) = ------ --> ---------------------
           s + wc      1 * s^0 + wc * s^-1

    If 'prewarped' is True, it is assumed that the provided cutoff frequency has
    already been adjusted based on the Bilinear Transform. If false, the
    frequency will be warped as part of this function's calculation.

    NOTE: 1st order filters don't generally have a 'Q' parameter.
    """
    wc = freq * math.pi * 2
    K  = bt.gen_K(freq, sample_rate, prewarped)
    return filter.Filter(coefficients=bt.general_difference_eq1(
        [0, wc],
        [1, wc],
        K
    ))

def butterworth_hp1(freq, sample_rate, prewarped=False):
    """
    Constructs a 1st order Butterworth highpass filter with the provided cutoff
    frequency (in Hz) based on the provided sample rate (in Hz) using the
    following transfer equation:

              s        1 * s^0 + 0 * s^-1
    H(s) = ------ --> ---------------------
           s + wc      1 * s^0 + wc * s^-1

    If 'prewarped' is True, it is assumed that the provided cutoff frequency has
    already been adjusted based on the Bilinear Transform. If false, the
    frequency will be warped as part of this function's calculation.

    NOTE: 1st order filters don't generally have a 'Q' parameter.

    NOTE: We can convert a lowpass filter to a highpass by replacing (s/wc) in
    the original specification with (wc/s). For example, a 1st order Butterworth
    is specified as:

             1                s
    H(a) = -----, where a = ----
           a + 1             wc

    If we use a = wc/s instead, we get the H(s) given above. For a general
    Butterworth, the high pass will have 0 coefficients for each element in the
    numerator except for the 1st term, which should be a '1'.
    """
    wc = freq * math.pi * 2
    K  = bt.gen_K(freq, sample_rate, prewarped)
    return filter.Filter(coefficients=bt.general_difference_eq1(
        [1, 0],
        [1, wc],
        K
    ))

# ---------------------------------------------------------------------------- #

def butterworth_lp2(freq, sample_rate, prewarped=False, q=1.0 / math.sqrt(2)):
    """
    Constructs a 2nd order Butterworth lowpass filter with the provided cutoff
    frequency (in Hz) based on the provided sample rate (in Hz) using the
    following transfer equation:

                   wc^2
    H(s) = ---------------------
           s^2 + wc/q * s + wc^2

    If 'prewarped' is True, it is assumed that the provided cutoff frequency has
    already been adjusted based on the Bilinear Transform. If false, the
    frequency will be warped as part of this function's calculation.

    Q can be adjusted to introduce resonance in the filter. Values larger than
    the default (1 / sqrt(2)) will provide more resonance, while smaller values
    will reduce the resonance. Note that filters with a non-default Q value
    aren't technically Butterworth filters anymore.
    """
    wc  = 2 * math.pi * freq        # cutoff frequency. 2Khz -> rads/second
    wc2 = wc * wc
    K   = bt.gen_K(freq, sample_rate, prewarped)
    return filter.Filter(coefficients=bt.general_difference_eq2(
        [0, 0, wc2],
        [1, wc / q, wc2],
        K
    ))

def butterworth_hp2(freq, sample_rate, prewarped=False, q=1.0 / math.sqrt(2)):
    """
    Constructs a 2nd order Butterworth highpass filter with the provided cutoff
    frequency (in Hz) based on the provided sample rate (in Hz) using the
    following transfer equation:

                   wc^2
    H(s) = ---------------------
           s^2 + wc/q * s + wc^2

    If 'prewarped' is True, it is assumed that the provided cutoff frequency has
    already been adjusted based on the Bilinear Transform. If false, the
    frequency will be warped as part of this function's calculation.

    Q can be adjusted to introduce resonance in the filter. Values larger than
    the default (1 / sqrt(2)) will provide more resonance, while smaller values
    will reduce the resonance. Note that filters with a non-default Q value
    aren't technically Butterworth filters anymore.
    """
    wc  = 2 * math.pi * freq        # cutoff frequency in rads/second
    wc2 = wc * wc
    K   = bt.gen_K(freq, sample_rate, prewarped)
    return filter.Filter(coefficients=bt.general_difference_eq2(
        [1, 0, 0],
        [1, wc/q, wc2],
        K
    ))

def butterworth_bp2(freq, sample_rate, prewarped=False, q=1.0 / math.sqrt(2)):
    """
    Constructs a 2nd order Butterworth bandpass filter with the provided center
    cutoff frequency (in Hz) based on the provided sample rate (in Hz) using the
    following transfer equation:

                 wc/q * s
    H(s) = ----------------------
           s^2 + wc/q * s + wc^2

    If 'prewarped' is True, it is assumed that the provided cutoff frequency has
    already been adjusted based on the Bilinear Transform. If false, the
    frequency will be warped as part of this function's calculation.

    Q determines the bandwidth of the filter. (We use a narrow band symmetric
    construction).

    The two edge frequencies can be recovered based on the following formulas:
            freq * (1 + sqrt(1 + 4 * Q^2))
      f2 = -------------------------------
                       2 * Q
           freq^2
      f1 = ------
             f2

    These are based on the following facts:
          freq
    Q = -------,       freq = sqrt(f1 * f2)
        f2 - f1

    The bandwidth can be calculated as f2 - f1 if needed. Alternatively, it is
    also equal to freq / Q.

    References:
      - https://www.mikroe.com/ebooks/digital-filter-design/analog-prototype-filter-to-analog-filter-transformation
      - https://www.analog.com/media/en/training-seminars/design-handbooks/Basic-Linear-Design/Chapter8.pdf
    """
    wc = 2 * math.pi * freq
    K  = bt.gen_K(freq, sample_rate, prewarped)

    return filter.Filter(coefficients=bt.general_difference_eq2(
        [0, wc/q, 0],
        [1, wc/q, wc * wc],
        K
    ))

def butterworth_notch2(freq, sample_rate, prewarped=False, q=1.0 / math.sqrt(2)):
    """
    Constructs a 2nd order Butterworth notch filter with the provided center
    cutoff frequency (in Hz) based on the provided sample rate (in Hz) using the
    following transfer equation:

                 s^2 + wc^2
    H(s) = ----------------------
           s^2 + wc/q * s + wc^2

    If 'prewarped' is True, it is assumed that the provided cutoff frequency has
    already been adjusted based on the Bilinear Transform. If false, the
    frequency will be warped as part of this function's calculation.

    Q determines the bandwidth of the filter. (We use a narrow band symmetric
    construction).
    """
    wc = 2 * math.pi * freq
    K  = bt.gen_K(freq, sample_rate, prewarped)

    return filter.Filter(coefficients=bt.general_difference_eq2(
        [1, 0, wc * wc],
        [1, wc/q, wc * wc],
        K
    ))

# ---------------------------------------------------------------------------- #

def butterworth_lp3(freq, sample_rate, prewarped=False):
    """
    Constructs a 3rd order Butterworth lowpass filter with the provided cutoff
    frequency (in Hz) based on the provided sample rate (in Hz) using the
    following transfer equation:

                             wc^3
    H(s) = ----------------------------------------
           s^3 + 2 * wc * s^2 + 2 * wc^2 * s + wc^3

    If 'prewarped' is True, it is assumed that the provided cutoff frequency has
    already been adjusted based on the Bilinear Transform. If false, the
    frequency will be warped as part of this function's calculation.
    """
    wc  = 2 * math.pi * freq        # cutoff frequency -> rads/second
    wc2 = wc * wc
    wc3 = wc * wc * wc
    K   = bt.gen_K(freq, sample_rate, prewarped)
    return filter.Filter(coefficients=bt.general_difference_eq3(
        [0, 0, 0, wc3],
        [1, 2 * wc, 2 * wc2, wc3],
        K
    ))

def butterworth_hp3(freq, sample_rate, prewarped=False):
    """
    Constructs a 3rd order Butterworth highpass filter with the provided cutoff
    frequency (in Hz) based on the provided sample rate (in Hz) using the
    following transfer equation:

                             s^3
    H(s) = ----------------------------------------
           s^3 + 2 * wc * s^2 + 2 * wc^2 * s + wc^3

    If 'prewarped' is True, it is assumed that the provided cutoff frequency has
    already been adjusted based on the Bilinear Transform. If false, the
    frequency will be warped as part of this function's calculation.
    """
    wc  = 2 * math.pi * freq        # cutoff frequency -> rads/second
    wc2 = wc * wc
    wc3 = wc * wc * wc
    K   = bt.gen_K(freq, sample_rate, prewarped)
    return filter.Filter(coefficients=bt.general_difference_eq3(
        [1, 0, 0, 0],
        [1, 2 * wc, 2 * wc2, wc3],
        K
    ))

# NOTE: There is no butterworth_bp3() or butterworth_notch3() because those
# filters only exist with even pole counts.

# ---------------------------------------------------------------------------- #

def butterworth_lp4(freq, sample_rate, prewarped=False):
    """
    Constructs a 4th order Butterworth lowpass filter with the provided cutoff
    frequency (in Hz) based on the provided sample rate (in Hz) using the
    following transfer equation:

                                                   wc^4
    H(s) = -------------------------------------------------------------------------------------
           s^4 + (c1 + c2) * wc * s^3 + (2 + c1 * c2) * wc^2 * s^2 + (c1 + c2) * wc^3 * s + wc^4

    where c1 = -2 * cos(5 * pi / 8),
          c2 = -2 * cos(7 * pi / 8)

    If 'prewarped' is True, it is assumed that the provided cutoff frequency has
    already been adjusted based on the Bilinear Transform. If false, the
    frequency will be warped as part of this function's calculation.
    """
    wc  = 2 * math.pi * freq
    wc2 = wc * wc
    wc3 = wc * wc * wc
    wc4 = wc * wc * wc * wc
    K   = bt.gen_K(freq, sample_rate, prewarped)
    c1  = -2 * math.cos(5 * math.pi / 8)
    c2  = -2 * math.cos(7 * math.pi / 8)

    return filter.Filter(coefficients=bt.general_difference_eq4(
        [0, 0, 0, 0, wc4],
        [1, (c1 + c2) * wc, (2 + c1 * c2) * wc2, (c1 + c2) * wc3, wc4],
        K
    ))

def butterworth_hp4(freq, sample_rate, prewarped=False):
    """
    Constructs a 4th order Butterworth highpass filter with the provided cutoff
    frequency (in Hz) based on the provided sample rate (in Hz) using the
    following transfer equation:

                                                   s^4
    H(s) = -------------------------------------------------------------------------------------
           s^4 + (c1 + c2) * wc * s^3 + (2 + c1 * c2) * wc^2 * s^2 + (c1 + c2) * wc^3 * s + wc^4

    where c1 = -2 * cos(5 * pi / 8), c2 = -2 * cos(7 * pi / 8)

    If 'prewarped' is True, it is assumed that the provided cutoff frequency has
    already been adjusted based on the Bilinear Transform. If false, the
    frequency will be warped as part of this function's calculation.
    """
    wc  = 2 * math.pi * freq
    wc2 = wc * wc
    wc3 = wc * wc * wc
    wc4 = wc * wc * wc * wc
    K   = bt.gen_K(freq, sample_rate, prewarped)
    c1  = -2 * math.cos(5 * math.pi / 8)
    c2  = -2 * math.cos(7 * math.pi / 8)

    return filter.Filter(coefficients=bt.general_difference_eq4(
        [1, 0, 0, 0, 0],
        [1, (c1 + c2) * wc, (2 + c1 * c2) * wc2, (c1 + c2) * wc3, wc4],
        K
    ))

def butterworth_bp4(freq, sample_rate, prewarped=False, q=1/math.sqrt(2)):
    """
    Constructs a 4th order Butterworth bandpass filter with the provided center
    cutoff frequency (in Hz) based on the provided sample rate (in Hz) using the
    following transfer equation:

                                        (wc/q)^2 * s^2
    H(s) = --------------------------------------------------------------------------------------
           s^4 + sqrt(2)/q * wc * s^3 + (2q^2 + 1)/q^2 * wc^2 * s^2 + sqrt(2)/q * wc^3 * s + wc^4

    If 'prewarped' is True, it is assumed that the provided cutoff frequency has
    already been adjusted based on the Bilinear Transform. If false, the
    frequency will be warped as part of this function's calculation.

    Q determines the bandwidth of the filter. (We use a narrow band symmetric
    construction).

    The two edge frequencies can be recovered based on the following formulas:
            freq * (1 + sqrt(1 + 4 * Q^2))
      f2 = -------------------------------
                       2 * Q
           freq^2
      f1 = ------
             f2

    These are based on the following facts:
          freq
    Q = -------,       freq = sqrt(f1 * f2)
        f2 - f1

    The bandwidth can be calculated as f2 - f1 if needed. Alternatively, it is
    also equal to freq / Q.
    """
    wc    = 2 * math.pi * freq
    wc2   = wc * wc
    wc3   = wc * wc * wc
    wc4   = wc * wc * wc * wc
    q2    = q * q
    root2 = math.sqrt(2)
    K     = bt.gen_K(freq, sample_rate, prewarped)

    return filter.Filter(coefficients=bt.general_difference_eq4(
        [0, 0, wc2 / q2, 0, 0],
        [1, root2/q * wc, (2 * q2 + 1) / q2 * wc2, root2 / q * wc3, wc4],
        K
    ))

def butterworth_notch4(freq, sample_rate, prewarped=False, q=1/math.sqrt(2)):
    """
    Constructs a 4th order Butterworth notch filter with the provided center
    cutoff frequency (in Hz) based on the provided sample rate (in Hz) using the
    following transfer equation:

                                    s^4 + 2 * wc^2 * s^2 + wc^4
    H(s) = ----------------------------------------------------------------------------------
           s^4 + sqrt(2) * wc/q * s^3 + (2q + 1)/q * wc^2 * s^2 + sqrt(2)/q * wc^3 * s + wc^4

    If 'prewarped' is True, it is assumed that the provided cutoff frequency has
    already been adjusted based on the Bilinear Transform. If false, the
    frequency will be warped as part of this function's calculation.

    Q determines the bandwidth of the filter. (We use a narrow band symmetric
    construction).
    """
    wc    = 2 * math.pi * freq
    wc2   = wc * wc
    wc3   = wc * wc * wc
    wc4   = wc * wc * wc * wc
    q2    = q * q
    root2 = math.sqrt(2)
    K     = bt.gen_K(freq, sample_rate, prewarped)

    return filter.Filter(coefficients=bt.general_difference_eq4(
        [1, 0, 2 * wc2, 0, wc4],
        [1, root2/q * wc, (2 * q + 1) / q * wc2, root2 / q * wc3, wc4],
        K
    ))

# ---------------------------------------------------------------------------- #

def cascaded_butterworth(poles, type, freq, sample_rate, prewarped=False):
    """
    Constructs a generic Butterworth filter with an arbitrary number of poles
    using the cascaded biquad technique. Essentially, ceil(poles/2) separate
    filters are created with differing Q values, and they each are applied in
    series on the provided signal. Filters designed using this technique are
    much less limited by filter order than the other implementations in this
    file, and they tend to be more stable in practice.

    'type' specifies which filter type to be used and must be one of
    ['lp', 'hp'].

    References:
      - https://www.earlevel.com/main/2016/09/29/cascading-filters/
      - https://en.wikipedia.org/wiki/Digital_biquad_filter
    """
    assert poles >= 1

    if type == 'lp':
        f1 = butterworth_lp1
        f2 = butterworth_lp2
    elif type == 'hp':
        f1 = butterworth_hp1
        f2 = butterworth_hp2
    else:
        assert false, "'type' must be one of [lp, hp]"

    filters = []

    # Calculate ideal interval between poles
    spacing = math.pi / poles

    # Handle odd pole count
    if poles % 2 == 1:
        # Handle 1-pole on real axis first
        filters.append(f1(freq, sample_rate, prewarped))

        # Handle all remaining poles. They should be equally spaced, at the
        # interval: 1 * angle, 2 * angle, etc.
        for i in range(poles//2):
            angle = (i + 1) * spacing
            q     = 1 / (2 * math.cos(angle))
            filters.append(f2(freq, sample_rate, prewarped, q=q))

    # Handle even pole count
    else:
        for i in range(poles//2):
            angle = spacing/2 + i * spacing
            q     = 1 / (2 * math.cos(angle))
            filters.append(f2(freq, sample_rate, prewarped, q=q))

    return filter.CascadedFilter(filters)
