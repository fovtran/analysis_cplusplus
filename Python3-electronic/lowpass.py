# low pass filter
def firfilt(interval, freq, sampling_rate):
    nfreq = freq/(0.5*sampling_rate)
    taps =  sampling_rate + 1
    a = 1
    b = scipy.signal.firwin(taps, cutoff=nfreq)
    return scipy.signal.lfilter(b, a, interval)
#fft
