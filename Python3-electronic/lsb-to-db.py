#!/usr/bin/env python

import math

"""
Zero-dB standards:

Audio industory:   0 dB = 1 mW in 600 Ohm resistance
      (Measurements to this standards use the unit symbol dBm)

Television industory:  0 dB = 1 mV rms across 75 Ohm
Radio frequency engineering:  0 dB = 1 mW in 50 Ohm resistance
	or 0 dB = 1 uV/m for electro-magnetic field strength
Radio engineers use absolute dBm of which zero-dB standard is 1 mW or absolute dBu of which zero-dB standard is 1uV.
"""

_vread = 1022.0
Vref = 5.0 # volts
depth = 12.0
_dmax = 2.0**depth
_epsilon = (Vref / _dmax)
mean_noise = _epsilon*_vread

# A = 10*math.log(P2/P1)  where P2 is input and P1 is reference
P2 = _epsilon
P1 = mean_noise
A = 20*math.log(P2/P1)

print ("Digital max is %8.2f v" % (_dmax))
print ("Digital unit is %.12f v" % (_epsilon))
print ("Mean noise is %12.12f uv" % (mean_noise))
print ("A =  %12.12f dB" % A)
