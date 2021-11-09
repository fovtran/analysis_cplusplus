import numpy as np
from matplotlib.pyplot import *

mod_noisy = .5
demod_1_a=mod_noisy*2*np.cos((2*np.pi*Fc*t)+phi)

N=10
Fc=40
Fs=1600
d=firwin(numtaps=N,cutoff=40,nyq=Fs/2)
print(len(d))
Hd=lfilter( d, 1.0, demod_1_a)
print(len(Hd))
y2=(convolve(Hd,raised))/Convfac
print(len(y2))
y2=y2[(sa/2)-1:-sa/2]
print(len(y2))
demod_3_a=y2[(sa/2)-1::sa]
print(len(demod_3_a))

demod_1_b=-1*mod_noisy*2*sin((2*pi*Fc*t)+phi)
Hd2=lfilter(d,1.0,demod_1_b)
y3=(convolve(Hd2,raised))/Convfac
y3=y3[(sa/2)-1:-sa/2]
demod_3_b=y3[(sa/2)-1::sa]

#########3333
#Demod

demod=demod_3_a+(1j)*demod_3_b
print((demod))
plot(demod,'wo')
plot(numpy.real(demod),'wo')
plot(numpy.imag(demod),'wo')

show()
