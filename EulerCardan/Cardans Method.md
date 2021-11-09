# Using Cardan’s method to solve Peng-Robinson equation
## may 5, 2018 by ali gabriel lara

# - Let’s assume a typical problem in Chemical Engineering Thermodynamics to show how to perform this kind of calculations using Python
import numpy as np

# PROBLEM STATEMENT
## Assuming that n-octane obeys the Peng-Robinson equation of state
## calculate the molar volumes of saturated liquid and saturated vapor at 427.85 K and 0.215 MPa.

## PENG-ROBINSON EQUATION
## This model is described for this set of equation

\displaystyle \begin{aligned} & \alpha = \left[1+ \left( 0.37464 +1.54226 \omega -0.26992 \omega^2\right) \left( 1 -\sqrt{Tr} \right)\right]^ 2 \\ & A = 0.45724\left(\frac{P_r}{T_r^2}\right)\alpha \\ & B = 0.07780 \left(\frac{P_r}{T_r}\right) \\ & Z^3 + (B-1) Z^2 + (A - 2B - 3B^2)Z - AB + B^2 + B^ 3 = 0 \end{aligned}

# Data from the problem
T = 427.85    # Kelvin degree
P = 0.215e6   # Pressure in Pa
Tc = 568.7    # Critical temperature, K
Pc = 24.90e5  # Critical pressure, Pa
omega = 0.4   # acentric factor
R = 8.314     # universal gas constant in (Pa m**3)/(mol K)
Let’s calculate the EOS parameters to build its Z-cubic polynomic
Pr = P / Pc
Tr = T / Tc
alpha =  ((1 + ( 0.37464 + 1.54226 * omega - 0.26992 * omega**2)
         * ( 1 - np.sqrt(Tr)))**2)
A = 0.45724 * (Pr / Tr**2) * alpha
B = 0.07780 * (Pr / Tr)

coef = np.zeros(4)
coef[0] = 1
coef[1] = B - 1
coef[2] = A - 2 * B - 3 * B**2
coef[3] = - A * B + B**2 + B**3

## CARDAN’S METHOD
## To determine the roots of a cubic equation. The cubic equation of state can be expressed as:

Z^3 + \alpha Z^2 + \beta Z + \gamma

Then the cardan coefficients would be

\displaystyle \begin{aligned} & p = \beta - \frac{{\alpha^2}}{3} \\ & q = \frac{{2\alpha^3}}{27} - \frac{{\alpha\,\beta}}{3} + \gamma \end{aligned}

giving the discriminant

\displaystyle D = \frac{{q^2}}{4} + \frac{{p^3}}{27}

If D > 0 , there is only one real root.

\displaystyle Z = \left(-\frac{q}{2} + \sqrt{D}\right)^{1/3} + \left(-\frac{q}{2} -\sqrt{D}\right)^{1/3} - \frac{{\alpha}}{3}

If D = 0 , there are three real roots and two of them are equal.

\displaystyle \begin{aligned} Z_1 =& -2\,\left( \frac{{q}}{2}\right)^{1/3} - \frac{{\alpha}}{3}\\ Z_2 = Z_3 =& \left( \frac{{q}}{2}\right)^{1/3} - \frac{{\alpha}}{3} \end{aligned}

If D < 0 , there are three unqueal real roots.

\displaystyle \begin{aligned} Z_1 =& 2\,r^{1/3}\,\cos\left(\frac{\theta}{3}\right) - \frac{{\alpha}}{3}\\ Z_2 = & 2\,r^{1/3}\,\cos\left(\frac{2\pi + \theta}{3}\right) - \frac{{\alpha}}{3}\\ Z_2 = & 2\,r^{1/3}\,\cos\left(\frac{4\pi + \theta}{3}\right) - \frac{{\alpha}}{3} \end{aligned}

where
\displaystyle \begin{aligned} & \cos \theta = \frac{{-q}}{2} \left( \frac{-27}{p^3} \right)^{1/2} \\ & r = \left( \frac{-p^3}{27} \right)^{1/2} \end{aligned}

This method can be written in a function like this one
def cardan(coef):

    aalpha = coef[1]
    beeta = coef[2]
    gaamma = coef[3]

    p = beeta - (aalpha**2) / 3
    q = 2 * aalpha**3 / 27 - aalpha * beeta / 3 + gaamma
    D = q**2 / 4 + p**3 / 27

    if D > 0:
            Z = ((-q/2 + np.sqrt(D))**(1/3) +
                (-q/2 - np.sqrt(D))**(1./3) - aalpha / 3)
    elif D == 0:
            Z1 = (-2* q / 2)**(1/3) - aalpha/3
            Z2 = (q / 2)**(1/3) - aalpha/3
            Z3 = Z2
            Z = [Z1 ,Z2, Z3]
    else:
            r = np.sqrt(- p**3 / 27)
            theta = np.arccos(-q / 2 * 1/r)
            # Calculations of theta should be in radians
            Z1 = 2 * (r**(1/3)) * np.cos( theta/3 ) - aalpha/3
            Z2 = (2 * (r**(1/3)) * np.cos((2 * np.pi + theta)/3)
                 - aalpha/3)
            Z3 = (2 * (r**(1/3)) * np.cos((4 * np.pi + theta)/3)
                 - aalpha/3)
            Z = [Z1, Z2, Z3]

    return Z
Now, let’s get the solution of the cubic polynomic
Zcardan = cardan(coef)

# Results
if np.size(Zcardan) == 3:
    vf = min(Zcardan) * R * T / P
    vg = max(Zcardan) * R * T / P

    print('Using Cardan''s method')
    print('The volume of n-octane saturated liquid = {:.3e} m**3/mol'.format(vf))
    print('The volume of n-octane saturated vapour = {:.3e} m**3/mol'.format(vg))
Using Cardans method
The volume of n-octane saturated liquid = 2.007e-04 m**3/mol
The volume of n-octane saturated vapour = 1.514e-02 m**3/mol
USING BUILT-IN FUNCTION “NUMPY.ROOTS”¶
This is very straigth-forward method
# Using Python functions
ZPR = np.roots(coef)
ZPR_real = ZPR[np.isreal(ZPR).real]
vPRf = min(ZPR_real ) * R * T / P
vPRg = max(ZPR_real ) * R * T / P
print('Using built-in functions')
print('The volume of n-octane saturated liquid = {:.3e} m**3/mol'.format(vPRf))
print('The volume of n-octane saturated vapour = {:.3e} m**3/mol'.format(vPRg))
Using built-in functions
The volume of n-octane saturated liquid = 2.007e-04 m**3/mol
The volume of n-octane saturated vapour = 1.514e-02 m**3/mol
