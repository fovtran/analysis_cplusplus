rate, data = wavfile.read('./TriLeftChannel.wav')

filtereddata = numpy.fft.rfft(data) # FFT Filtered data
freqdata = numpy.fft.fftfreq(data.size) #Frequency data
filtereddata = AudioFunctions.Filter(filtereddata, freqdata, data, rate) # Filter the data

def Filter(filtereddata, freqdata, data, rate):

    #fftchunks = (rate / 2)# + 1

    x = freqdata[:len(data) / 2]

    for f in range(len(x)):

        if x[f] > 0.1:
            filtereddata[f] = 0.0
            filtereddata[len(data) / 2 + f] = 0.0

    return filtereddata

filteredwrite = numpy.fft.irfft(filtereddata)

filteredwrite = numpy.round(filteredwrite).astype('int16') # Round off the numbers, and get ready to save it as 16-bit depth file (?)

wavfile.write('TestFiltered.wav', rate, filteredwrite)

def firfilt(interval, freq, sampling_rate):
    nfreq = freq/(0.5*sampling_rate)
    taps =  sampling_rate + 1
    a = 1
    b = scipy.signal.firwin(taps, cutoff=nfreq)
    return scipy.signal.lfilter(b, a, interval)


Hi,

Is there a direct equivalent of matlab's impz() function in scipy?
In matlab I can do:

[b,a] = butter(4,[1.5/678,15.0/678],'bandpass');
[h,t] = impz(b,a);
plot(t,h)

The scipy.signal.impulse function is the closest I can find, but

b,a = scipy.signal.butter(4,[1.5/678,15.0/678],'bandpass')
T,h = scipy.signal.impulse((b,a))
plot(T,h)

doesn't give the same answer.

Will
