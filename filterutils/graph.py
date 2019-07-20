import math
import numpy as np
import matplotlib.pyplot as plt

"""
This file contains several helper functions that can be used to graph filter
attributes using matplotlib. These help to verify that the filters are designed
correctly and can be used to examine the behavior of new filters.
"""

def plot_frequency_response(responses, sample_rate,
    amp_min=-80, amp_max=10,
    xscale='log',
    freq_min=None, freq_max=None
    ):

    """
    Plots the given frequency response curves (both amplitude and phase)

    Parameters:
      responses (list)     - List of responses (in format defined by
          frequency_response())
      sample_rate (number) - Chosen sample rate
      amp_min (number)     - Min amplitude (in dB)     - defaults to '-80'
      amp_max (number)     - Max amplitude (in dB)     - defaults to '10'
      xscale  (string)     - One of ['linear', 'log']  - defaults to 'log'
      freq_min (number)    - Minimum frequency (in Hz) - defaults to 1
      freq_max (number)    - Maximum frequency (in Hz) - defaults to sample_rate / 2
    """
    assert isinstance(responses, (list, tuple, np.ndarray)) \
        and isinstance(responses[0], np.ndarray) \
        and responses[0].dtype == np.complex128, "Responses must be list of np.complex128 arrays"

    # Handle min/max frequency options
    if freq_min is None:
       freq_min = 1
    else:
        freq_min = max(1, freq_min)

    if freq_max is None:
        freq_max = sample_rate / 2
    else:
        freq_max = min(freq_max, sample_rate / 2)

    # X axis (frequency)
    N     = len(responses[0])
    freqs = [(sample_rate / 2) * i / N for i in range(N)]

    # Y axes (amplitude in db and phase in radians)
    plt.figure()
    plt.subplot(211)
    plt.ylabel('Amplitude (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.xscale(xscale)
    plt.axis([freq_min, freq_max, amp_min, amp_max])
    plt.grid(axis='both')

    plt.subplots_adjust(top=0.95, hspace=0.3)

    plt.subplot(212)
    plt.ylabel('Phase (Radians)')
    plt.xlabel('Frequency (Hz)')
    plt.axis([freq_min, freq_max, -math.pi, math.pi])
    plt.xscale(xscale)
    plt.grid(axis='both')

    for response in responses:
        mag   = np.abs(response)
        mag   = [20 * math.log10(s + 1E-17) for s in mag ] # epsilon to prevent log(0)
        phase = np.angle(response)

        plt.subplot(211)
        plt.plot(freqs, mag)
        plt.subplot(212)
        plt.plot(freqs, phase)
    plt.show()

def plot_impulse_response(responses):
    plt.figure()
    plt.ylabel('Amplitude')
    plt.xlabel('Samples (n)')

    for response in responses:
        plt.plot(response)
    plt.show()
