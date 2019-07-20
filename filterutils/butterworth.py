from . import filter
import cmath

"""
The functions in this file are used to design Butterworth filters of any
arbitrary order in one of the four basic forms:
  - Lowpass
  - Highpass
  - Bandpass
  - Notch (Bandstop)

Each function allows the caller to add resonance to the filter by specifying a
value between [-1, 1] for the 'resonance' argument. Positive resonance
introduces a gain > 1 at or near the cutoff frequency, while negative resonance
softens the filter.

Each function can be used to return either a single filter (with a potentially
large number of coefficients) or a cascaded biquad filter, in which each
individual filter has an order of at most 2, but several filters may be applied
in sequence.

These functions work by directly calculating the poles and zeroes of the
corresponding analog filter, transforming them to the z-domain via the Bilinear
Transform, and multiplying them together to form the numerator (zeroes) and
denominator (poles) of the digital transfer function H(z). The process is a bit
more involved than that used in butterworth_basic.py, but the results are much
more general.
"""

def butterworth_analog_poles(N, freq, sample_rate, resonance=0.0, mode='lp'):
    """
    Calculates the N (or 2N) complex poles for a Butterworth filter with
    order N using the provided cutoff frequency (in Hz) and sample rate (in Hz).
    The poles will be returned based on the provided mode.

    Modes:
      - 'lp' - Lowpass  - N poles on the left side of the s-plane
      - 'hp' - Highpass - N poles on the right side of the s-plane
    """
    assert freq <= sample_rate/2, "Freq must be less than sample_rate/2"

    # 1a. Identify the poles of the generic analog Butterworth LP filter. They
    # should be evenly spaced around the s = -1 + 0i line on a circle of unit
    # radius.
    theta = [ (2 * k - 1) * cmath.pi / (2 * N) for k in range(1, N + 1) ]

    # 1b. Apply resonance. We add a small delta to the first and last list
    # elements, which correspond to the angles nearest the imaginary axis.
    delta      = resonance * cmath.pi / (2 * N)
    theta[0]  -= delta
    theta[-1] += delta

    # Compute the pole locations in s-space before the frequency cutoff has
    # been provided
    pa = [ complex(-cmath.sin(t), cmath.cos(t)) for t in theta]

    # 2. Prewarp the digital frequency to get a corresponding analog frequency
    # (part of the Bilinear Transform). Use that to correctly scale each of the
    # LP poles calculated in step 1.
    freq_analog = sample_rate / cmath.pi * cmath.tan(cmath.pi * freq / sample_rate)

    # 3. Apply the proper transform to ensure the correct poles are returned for
    # each mode
    if mode == 'lp':
        return [ 2 * cmath.pi * freq_analog * p for p in pa ]
    elif mode == 'hp':
        return [ 2 * cmath.pi * freq_analog / p for p in pa ]
    else:
        assert false, "Mode must be one of [lp, hp]"

def butterworth_lp(N, freq, sample_rate, resonance=0.0, cascade=False):
    """
    Constructs an Nth order Butterworth lowpass filter with the provided cutoff
    frequency (in Hz) based on the provided sample rate (in Hz) using the direct
    method.

    The basic steps are:
    1) Generate the N poles of the analog Butterworth lowpass filter.
    2) Apply the Bilinear Transform to each individual pole to convert from the
       s domain to the z domain.
    3) Use the poles (and zeroes, which are all -1 in the z domain) to calculate
       the coefficients of the numerator and denominator polynomials for the
       transfer function H(z). Those coefficients are used directly for the
       filter's difference equation.

    NOTES:
    1) When resonance is > 0, the frequency response will spike at the provided
       cutoff frequency. When resonance < 0, the frequency response drops with
       a shallower slope. Resonance is achieved by rotating the poles nearest
       the imaginary axis in s-space closer towards the imaginary axis while
       leaving the other poles at their original locations. Resonance should be
       limited to the range [-1, 1] for best effect.

    References:
      - https://www.dsprelated.com/showarticle/1119.php
    """

    # 1. Compute the poles for the analog Butterworth of the proper order
    pa = butterworth_analog_poles(N, freq, sample_rate, resonance=resonance, mode='lp')

    # 2. Apply the Bilinear Transform to each analog pole to convert from
    # the s domain to the z domain. Our zeroes will all be -1 (by the nature
    # of the Butterworth LP formulation)
    #
    #       1 + (pa / 2 * sample_rate)
    # p_d = --------------------------
    #       1 - (pa / 2 * sample_rate)
    #
    pd = [ (1 + p / (2 * sample_rate)) / (1 - p / (2 * sample_rate)) for p in pa ]
    qd = [ -1 for i in range(N) ]

    # 3. Convert poles and zeroes to a transfer equation H(z) that can be used
    # to determine the difference equation coefficients. We want the gain at
    # z = e^(i * w), w = 0 to equal 1. In the case of a lowpass filter, this
    # is simply 1 + 0i.
    return filter.create_digital_filter(pd, qd, z0=1, cascade=cascade)

def butterworth_hp(N, freq, sample_rate, resonance=0.0, cascade=False):
    """
    Constructs an Nth order Butterworth highpass filter with the provided cutoff
    frequency (in Hz) based on the provided sample rate (in Hz) using the direct
    method.

    The basic steps are:
    1) Generate the N poles of the analog Butterworth highpass filter.
    2) Apply the Bilinear Transform to each individual pole to convert from the
       s domain to the z domain.
    3) Use the poles (and zeroes, which are all +1 in the z domain) to calculate
       the coefficients of the numerator and denominator polynomials for the
       transfer function H(z). Those coefficients are used directly for the
       filter's difference equation.

    NOTES:
    1) When resonance is > 0, the frequency response will spike at the provided
       cutoff frequency. When resonance < 0, the frequency response drops with
       a shallower slope. Resonance is achieved by rotating the poles nearest
       the imaginary axis in s-space closer towards the imaginary axis while
       leaving the other poles at their original locations. Resonance should be
       limited to the range [-1, 1] for best effect.

    References:
      - https://www.dsprelated.com/showarticle/1135.php
    """
    # 1. Compute the poles for the analog Butterworth of the proper order
    pa = butterworth_analog_poles(N, freq, sample_rate, resonance=resonance, mode='hp')

    # 2. Apply the Bilinear Transform to each individual pole to convert from
    # the s domain to the z domain. Our zeroes will all be +1 (by the nature
    # of the Butterworth HP formulation)
    #
    #       1 + (pa / 2 * sample_rate)
    # p_d = --------------------------
    #       1 - (pa / 2 * sample_rate)
    #
    pd = [ (1 + p / (2 * sample_rate)) / (1 - p / (2 * sample_rate)) for p in pa ]
    qd = [ 1 for i in range(N) ]

    # 3. Convert poles and zeroes to a transfer equation H(z) that can be used
    # to determine the difference equation coefficients. We want the gain at
    # z = e^(i * w), w = pi to equal 1. In the case of a highpass filter, this
    # is simply -1 + 0i.
    return filter.create_digital_filter(pd, qd, z0=-1, cascade=cascade)

def butterworth_bp(N, freq, bandwidth_hz, sample_rate, resonance=0.0, cascade=False):
    """
    Constructs an 2 * Nth order Butterworth bandpass filter with the provided
    center frequency (in Hz) based on the provided sample rate (in Hz) using
    the direct method based on a Butterworth lowpass filter of order N.

    NOTES:
    1) When resonance is > 0, the frequency response will spike around the provided
       center frequency (rather than at the center frequency like the lowpass
       or highpass filters). When resonance < 0, the frequency response drops with
       a shallower slope. Resonance is achieved by rotating the poles nearest
       the imaginary axis in s-space closer towards the imaginary axis while
       leaving the other poles at their original locations. Resonance should be
       limited to the range [-1, 1] for best effect.

    References:
      - https://www.dsprelated.com/showarticle/1128.php
    """
    f1 = freq - bandwidth_hz / 2
    f2 = freq + bandwidth_hz / 2
    assert f2 < sample_rate / 2, "Upper frequency must be less than sample_rate/2"
    assert f1 > 0, "Lower frequency must be greater than 0"

    # 1a. Identify the poles of the generic analog Butterworth LP filter. They
    # should be evenly spaced around the s = -1 + 0i line on a circle of unit
    # radius.
    theta = [ (2 * k - 1) * cmath.pi / (2 * N) for k in range(1, N + 1) ]

    # 1b. Apply resonance. We add a small delta to the first and last list
    # elements, which correspond to the angles nearest the imaginary axis.
    delta      = resonance * cmath.pi / (2 * N)
    theta[0]  -= delta
    theta[-1] += delta

    # Compute the pole locations in s-space before the frequency cutoff has
    # been provided
    p_lp = [ complex(-cmath.sin(t), cmath.cos(t)) for t in theta]

    # 2. Prewarp f0, f1, and f2 to get their analog equivalents
    F1    = sample_rate / cmath.pi * cmath.tan(cmath.pi * f1 / sample_rate)
    F2    = sample_rate / cmath.pi * cmath.tan(cmath.pi * f2 / sample_rate)
    BW_hz = F2 - F1
    F0    = cmath.sqrt(F1 * F2)

    # 3. Transform each of the analog lp poles into 2 analog bp poles
    alpha = [ (BW_hz * p) / (2 * F0) for p in p_lp ]
    beta  = [ cmath.sqrt(1 - p ** 2) for p in alpha ]
    com   = 2 * cmath.pi * F0
    pa    = [ com * (alpha[i] + 1j * beta[i]) for i in range(N) ] + \
            [ com * (alpha[i] - 1j * beta[i]) for i in range(N) ]

    # 4. Calculate the digital poles and zeroes - We apply the Bilinear
    # transform to the poles and the zeroes will be at +/- 1.
    #       1 + (pa / 2 * sample_rate)
    # p_d = --------------------------
    #       1 - (pa / 2 * sample_rate)
    pd = [ (1 + p / (2 * sample_rate)) / (1 - p / (2 * sample_rate)) for p in pa ]
    qd = [-1] * N + [1] * N

    # 5. Convert poles and zeroes to a transfer equation H(z) that can be used
    # to determine the difference equation coefficients. We want the gain at
    # z = e^(i * w), w = 2 * pi * (f0 / sample_rate) to equal 1.
    f0 = cmath.sqrt(f1 * f2)
    z0 = cmath.exp(1j * 2 * cmath.pi * (f0 / sample_rate))
    return filter.create_digital_filter(pd, qd, z0=z0, cascade=cascade)

def butterworth_notch(N, freq, bandwidth_hz, sample_rate, resonance=0.0, cascade=False):
    """
    Constructs an 2 * Nth order Butterworth notch filter with the provided
    center frequency (in Hz) based on the provided sample rate (in Hz) using
    the direct method based on a Butterworth lowpass filter of order N.

    NOTES:
    1) When resonance is > 0, the frequency response will spike around the provided
       center frequency (rather than at the center frequency like the lowpass
       or highpass filters). When resonance < 0, the frequency response drops with
       a shallower slope. Resonance is achieved by rotating the poles nearest
       the imaginary axis in s-space closer towards the imaginary axis while
       leaving the other poles at their original locations. Resonance should be
       limited to the range [-1, 1] for best effect.

    References:
      - https://www.dsprelated.com/showarticle/1131.php
    """
    f1 = freq - bandwidth_hz / 2
    f2 = freq + bandwidth_hz / 2
    assert f2 < sample_rate / 2, "Upper frequency must be less than sample_rate/2"
    assert f1 > 0, "Lower frequency must be greater than 0"

    # 1a. Identify the poles of the generic analog Butterworth LP filter. They
    # should be evenly spaced around the s = -1 + 0i line on a circle of unit
    # radius.
    theta = [ (2 * k - 1) * cmath.pi / (2 * N) for k in range(1, N + 1) ]

    # 1b. Apply resonance. We add a small delta to the first and last list
    # elements, which correspond to the angles nearest the imaginary axis.
    delta      = resonance * cmath.pi / (2 * N)
    theta[0]  -= delta
    theta[-1] += delta

    # Compute the pole locations in s-space before the frequency cutoff has
    # been provided
    p_lp = [ complex(-cmath.sin(t), cmath.cos(t)) for t in theta]

    # 2. Prewarp f0, f1, and f2 to get their analog equivalents
    F1    = sample_rate / cmath.pi * cmath.tan(cmath.pi * f1 / sample_rate)
    F2    = sample_rate / cmath.pi * cmath.tan(cmath.pi * f2 / sample_rate)
    BW_hz = F2 - F1
    F0    = cmath.sqrt(F1 * F2)

    # 3. Transform each of the analog lp poles into 2 analog bp poles
    alpha = [ BW_hz / (2 * F0 * p) for p in p_lp ]
    beta  = [ cmath.sqrt(1 - p ** 2) for p in alpha ]
    com   = 2 * cmath.pi * F0
    pa    = [ com * (alpha[i] + 1j * beta[i]) for i in range(N) ] + \
            [ com * (alpha[i] - 1j * beta[i]) for i in range(N) ]

    # 4. Calculate the digital poles and zeroes - We apply the Bilinear
    # transform to the poles and the zeroes will be at e^(i * w) and e^(-i * w),
    # where w = 2 * pi * freq / sample_rate
    #       1 + (pa / 2 * sample_rate)
    # p_d = --------------------------
    #       1 - (pa / 2 * sample_rate)
    pd = [ (1 + p / (2 * sample_rate)) / (1 - p / (2 * sample_rate)) for p in pa ]
    w  = 2 * cmath.pi * freq / sample_rate
    qd = [ cmath.exp( 1j * w) for i in range(N) ] + \
         [ cmath.exp(-1j * w) for i in range(N) ]

    # 5. Convert poles and zeroes to a transfer equation H(z) that can be used
    # to determine the difference equation coefficients. We want the gain at
    # z = e^(i * w), w = 0 to equal 1. In the case of a notch filter, this
    # is simply 1 + 0i.
    return filter.create_digital_filter(pd, qd, z0=1, cascade=cascade)
