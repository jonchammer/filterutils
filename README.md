# Summary
This project contains several functions and classes related to DSP, especially
filtering. The most important classes are `Filter` and `CascadedFilter`, which
represent a single filter and a cascade of filters (series interpretation),
respectively. Filters are represented by the coefficients of their transfer
function, H(z), from which the transfer function can be interpreted directly.

Implementations are provided for Butterworth lowpass, highpass, bandpass,
and notch filters of any order. Any of these implementations can be realized
directly or as a cascaded biquad. In addition, each implementation includes a
resonance control that makes the filters suitable for audio applications.

# External Dependencies
1. Numpy
2. Matplotlib (used only for graph.py)

# Example Usage
```python
import filterutils

N           = 4       # Filter order
freq        = 1000    # Cutoff frequency in Hertz
sample_rate = 44100   # Sample rate in Hertz

# Design an Nth order butterworth lowpass filter
f = filterutils.butterworth_lp(N, freq, sample_rate, resonance=0.0, cascade=False)

# Plot and show the impulse response (samples vs. amplitude)
filterutils.plot_impulse_response(
    [
        f.impulse_response(512)
    ]
)

# Plot and show the frequency response (frequency vs. gain and frequency vs. phase)
filterutils.plot_frequency_response(
    [
        f.frequency_response(512)
    ],
    sample_rate, amp_max=20
)

# Apply the filter to some signal
signal = [ i for i in range(512) ]
output, _, _ = f.apply(signal)

# ... or apply the filter in batches
output_0, x_hist, y_hist = f.apply(signal[0:256])
output_1, _, _           = f.apply(signal[256:], x_hist, y_hist)
output = output_0 + output_1
```
