import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_colormap(seq, add_first_last=True, alpha=False,
                  name='CustomMap', register=True, N=256):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).

    FROM: http://stackoverflow.com/questions/16834861/
    Adapted with help from:
    http://matplotlib.org/examples/pylab_examples/custom_cmap.html
    """
    if add_first_last:
        seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    if alpha:
        cdict['alpha']=[]
    for i, item in enumerate(seq):
        if isinstance(item, float):
            if alpha:
                r1, g1, b1, a1 = seq[i - 1]
                r2, g2, b2, a2 = seq[i + 1]
                cdict['alpha'].append([item, a1, a2])
            else:
                r1, g1, b1 = seq[i - 1]
                r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    colormap = mcolors.LinearSegmentedColormap(name, cdict, N=N)
    if register:
        plt.register_cmap(cmap=colormap)
    return colormap

# Convert wavelength to color
def lin_inter(value, left=0., right=1., increase=True):
    """
    Returns the fractional position of ``value`` between ``left`` and
    ``right``, increasing from 0 if ``value==left`` to 1 if ``value==right``,
    or decreasing from 1 to zero if not ``increase``.
    """
    if increase:
        return (value - left) / (right - left)
    else:
        return (right - value) / (right - left)

def wav2RGB(Wavelength, upto255=False, Gamma=1.0):
    """
    Converts an wavelength to an RGB list, with fractional values between
    0 and 1 if not ``upto255``, or int values between 0 and 255 if ``upto255``
    """
    # Check the interval the color is in
    if 380 <= Wavelength < 440:
        # red goes from 1 to 0:
        Red = lin_inter(Wavelength, 380., 440., False)
        Green = 0.0
        Blue = 1.0
    elif 440 <= Wavelength < 490:
        Red = 0.0
        # green goes from 0 to 1:
        Green = lin_inter(Wavelength, 440., 490., True)
        Blue = 1.0
    elif 490 <= Wavelength < 510:
        Red = 0.0
        Green = 1.0
        Blue = lin_inter(Wavelength, 490., 510., False)
    elif 510 <= Wavelength < 580:
        Red = lin_inter(Wavelength, 510., 580., True)
        Green = 1.0
        Blue = 0.0
    elif 580 <= Wavelength < 645:
        Red = 1.0
        Green = lin_inter(Wavelength, 580., 645., False)
        Blue = 0.0
    elif 645 <= Wavelength <= 780:
        Red = 1.0
        Green = 0.0
        Blue = 0.0
    else: # Wavelength < 380 or Wavelength > 780
        Red = 0.0
        Green = 0.0
        Blue = 0.0
    # Let the intensity fall off near the vision limits
    if 380 <= Wavelength < 420:
        factor = 0.3 + 0.7*lin_inter(Wavelength, 380., 420., True)
    elif 420 <= Wavelength < 700:
        factor = 1.0
    elif 700 <= Wavelength <= 780:
        factor = 0.3 + 0.7*lin_inter(Wavelength, 700., 780., False)
    else:
        factor = 0.0
    # Adjust color intensity
    if upto255:
        def Adjust(Color, Factor):
            return int(round(255. * (Color * Factor)**Gamma))
    else:
        def Adjust(Color, Factor):
            return (Color * Factor)**Gamma
    R = Adjust(Red, factor)
    G = Adjust(Green, factor)
    B = Adjust(Blue, factor)
    #return color
    return [R, G, B]

# Create colormap based on function above:
wl_breakpoints = [380, 420, 440, 490, 510, 580, 645,700,780]
breakpoints = [lin_inter(point, 380., 780.) for point in wl_breakpoints]
colors = [wav2RGB(point) for point in wl_breakpoints]
N = len(breakpoints)
seq = []
for i in xrange(N):
    seq.extend((colors[i], breakpoints[i], colors[i]))

make_colormap(seq=seq, add_first_last=False, alpha=False,
              name='CustomSpectrum', register=True)

# Draw colormaps
def compare_spectrums(cmaps=[], n_points=256):
    """
    adapted from:
    http://matplotlib.org/1.3.0/examples/color/colormaps_reference.html
    """
    nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
    gradient = np.linspace(0, 1, n_points)
    gradient = np.vstack((gradient, gradient))

    def plot_color_gradients(cmap_category, cmap_list):
        fig, axes = plt.subplots(nrows=nrows)
        fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
        axes[0].set_title(cmap_category + ' colormaps', fontsize=14)
        for ax, name in zip(axes, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3]/2.
            fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

        #Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axes:
           ax.set_axis_off()

    for cmap_category, cmap_list in cmaps:
        plot_color_gradients(cmap_category, cmap_list)
    plt.show()

if __name__ == '__main__':
    # Create random spectrum
    n_points = 1000
    wavelengths = np.linspace(380, 780, n_points)
    spectrum = np.random.random(n_points)
    def color_and_intensity(wavelength):
        return [c*np.interp(x=wavelength, xp=wavelengths, fp=spectrum)\
                for c in wav2RGB(wavelength)]
    colors = [color_and_intensity(point) for point in wavelengths]
    breakpoints = [lin_inter(point, 380., 780.) for point in wavelengths]
    N = len(breakpoints)
    seq = []
    for i in xrange(N):
        seq.extend((colors[i], breakpoints[i], colors[i]))

    make_colormap(seq=seq, add_first_last=False, alpha=False,
                  name='RandomSpectrum', register=True, N=n_points)

    cmaps = [('Spectrum', ['spectral','gist_rainbow',
                           'CustomSpectrum','RandomSpectrum'])]
    compare_spectrums(cmaps=cmaps, n_points=n_points)
