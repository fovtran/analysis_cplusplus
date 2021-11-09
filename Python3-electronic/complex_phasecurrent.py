def mainWithNativeComplex():
    pSource = 750*e**(1j*pi/6)# power source, in polar form
                              # with a magnitude of 750 volts, and angle of 30
                              # degrees (pi/6 radians).
    r = 90                    # Ohms ( real part only )
    L = 0+160j                # Ohms ( Cartesian form )
    C = 0-40j                 # Ohms ( Cartesian form )
    Z = r+L+C                 # total impedance
    print "Impedance is ", Z, "(mag=", abs(Z), "; phase=", m.degrees(m.atan2(Z.imag, Z.real)), ")"

    I = pSource / Z
    print "Phase current is ", I, "(mag=", abs(I), "; phase=", m.degrees(m.atan2(I.imag, I.real)), ")"