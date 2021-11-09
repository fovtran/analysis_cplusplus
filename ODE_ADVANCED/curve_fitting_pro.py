import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # for unequal plot boxes
import scipy.optimize

# define fitting function
def GaussPolyBase(f, a, b, c, P, fp, fw):
    return a + b*f + c*f*f + P*np.exp(-0.5*((f-fp)/fw)**2)

# read in spectrum from data file
# f=frequency, s=signal, ds=s uncertainty
f, s, ds = np.loadtxt("Spectrum.txt", skiprows=4, unpack=True)

# initial guesses for fitting parameters
a0, b0, c0 = 60., -3., 0.
P0, fp0, fw0 = 80., 11., 2.

# fit data using SciPy's Levenberg-Marquart method
nlfit, nlpcov = scipy.optimize.curve_fit(GaussPolyBase,
                f, s, p0=[a0, b0, c0, P0, fp0, fw0], sigma=ds)

# unpack fitting parameters
a, b, c, P, fp, fw = nlfit
# unpack uncertainties in fitting parameters from diagonal
# of covariance matrix
da, db, dc, dP, dfp, dfw = \
          [np.sqrt(nlpcov[j,j]) for j in range(nlfit.size)]

# create fitting function from fitted parameters
f_fit = np.linspace(0.0, 25., 128)
s_fit = GaussPolyBase(f_fit, a, b, c, P, fp, fw)

# Calculate residuals and reduced chi squared
resids = s - GaussPolyBase(f, a, b, c, P, fp, fw)
redchisqr = ((resids/ds)**2).sum()/float(f.size-6)

# Create figure window to plot data
fig = plt.figure(1, figsize=(8,8))
gs = gridspec.GridSpec(2, 1, height_ratios=[6, 2])

# Top plot: data and fit
ax1 = fig.add_subplot(gs[0])
ax1.plot(f_fit, s_fit)
ax1.errorbar(f, s, yerr=ds, fmt='or', ecolor='black')
ax1.set_xlabel('frequency (THz)')
ax1.set_ylabel('absorption (arb units)')
ax1.text(0.7, 0.95, 'a = {0:0.1f}$\pm${1:0.1f}'
         .format(a, da), transform = ax1.transAxes)
ax1.text(0.7, 0.90, 'b = {0:0.2f}$\pm${1:0.2f}'
         .format(b, db), transform = ax1.transAxes)
ax1.text(0.7, 0.85, 'c = {0:0.2f}$\pm${1:0.2f}'
         .format(c, dc), transform = ax1.transAxes)
ax1.text(0.7, 0.80, 'P = {0:0.1f}$\pm${1:0.1f}'
         .format(P, dP), transform = ax1.transAxes)
ax1.text(0.7, 0.75, 'fp = {0:0.1f}$\pm${1:0.1f}'
         .format(fp, dfp), transform = ax1.transAxes)
ax1.text(0.7, 0.70, 'fw = {0:0.1f}$\pm${1:0.1f}'
         .format(fw, dfw), transform = ax1.transAxes)
ax1.text(0.7, 0.60, '$\chi_r^2$ = {0:0.2f}'
         .format(redchisqr),transform = ax1.transAxes)
ax1.set_title('$s(f) = a+bf+cf^2+P\,e^{-(f-f_p)^2/2f_w^2}$')

# Bottom plot: residuals
ax2 = fig.add_subplot(gs[1])
ax2.errorbar(f, resids, yerr = ds, ecolor="black", fmt="ro")
ax2.axhline(color="gray", zorder=-1)
ax2.set_xlabel('frequency (THz)')
ax2.set_ylabel('residuals')
ax2.set_ylim(-20, 20)
ax2.set_yticks((-20, 0, 20))

plt.show()
