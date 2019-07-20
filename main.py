import filterutils

def main():
    freq         = 1000
    bandwidth_hz = 400
    sample_rate  = 44100
    steps        = 512

    f0 = filterutils.butterworth_lp(4, freq, sample_rate, resonance=0.0, cascade=False)
    # f1 = b.butterworth_lp(4, freq, sample_rate, resonance=0.85, cascade=True)
    # f2 = b.butterworth_lp(5, freq, sample_rate, cascade=True)

    f1 = filterutils.butterworth_hp(4, freq, sample_rate, resonance=0.0, cascade=False)
    # f1 = b.butterworth_hp(4, freq, sample_rate, resonance=0.0, cascade=True)
    # f2 = b.butterworth_hp(5, freq, sample_rate, cascade=True)

    f2 = filterutils.butterworth_bp(2, freq, bandwidth_hz, sample_rate, resonance=0.0, cascade=False)
    # f1 = b.butterworth_bp(4, freq, bandwidth_hz, sample_rate, resonance=0.85, cascade=True)
    # f2 = b.butterworth_bp(9, freq, bandwidth_hz, sample_rate, cascade=True)

    f3 = filterutils.butterworth_notch(2, freq, bandwidth_hz, sample_rate, resonance=0.0, cascade=False)
    # f1 = b.butterworth_notch(2, freq, bandwidth_hz, sample_rate, resonance=0.85, cascade=True)
    # f2 = b.butterworth_notch(3, freq, bandwidth_hz, sample_rate, resonance=0.0, cascade=True)

    filterutils.plot_impulse_response(
        [
            f0.impulse_response(steps),
            f1.impulse_response(steps),
            f2.impulse_response(steps),
            f3.impulse_response(steps)
        ]
    )
    filterutils.plot_frequency_response(
        [
            f0.frequency_response(steps, fft=False),
            f1.frequency_response(steps, fft=False),
            f2.frequency_response(steps, fft=False),
            f3.frequency_response(steps, fft=False)
        ],
        sample_rate, amp_max=20
    )

if __name__ == '__main__':
    main()
