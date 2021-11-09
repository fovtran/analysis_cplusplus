def generateColor(color):
  nstep = 300
  minW = 400
  maxW = 700
  bandW = maxW - minW
  colorTuple = ()
  for i in range(nstep + 1):
    wlength = minW + i * bandW / nstep
    colorTuple += ((1.0 * i / nstep, wav2RGB(wlength)[color] / 255.0, wav2RGB(wlength)[color] / 255.0),)
  return colorTuple

def generateGradient():
  return {'red':  generateColor(0),
          'green': generateColor(1),
          'blue':  generateColor(2)}

visibleSpec = LinearSegmentedColormap('visible', generateGradient())

def wav2RGB(Wavelength):
    Gamma = 0.80
    IntensityMax = 255.0
    def Adjust(Color, Factor):
        if Color == 0.0:
            return 0.0
        else:
            return round(IntensityMax * (Color * Factor)**Gamma)
    if 380 <= trunc(Wavelength) and trunc(Wavelength) <= 439:
        Red = -(Wavelength - 440.0) / (440.0 - 380.0)
        Green = 0.0
        Blue = 1.0
    elif 440 <= trunc(Wavelength) and trunc(Wavelength) <= 489:
        Red = 0.0
        Green = (Wavelength - 440.0) / (490.0 - 440.0)
        Blue = 1.0
    elif 490 <= trunc(Wavelength) and trunc(Wavelength) <= 509:
        Red = 0.0
        Green = 1.0
        Blue = -(Wavelength - 510.0) / (510.0 - 490.0)
    elif 510 <= trunc(Wavelength) and trunc(Wavelength) <= 579:
        Red = (Wavelength - 510.0) / (580.0 - 510.0)
        Green = 1.0
        Blue = 0.0
    elif 580 <= trunc(Wavelength) and trunc(Wavelength) <= 644:
        Red = 1.0
        Green = -(Wavelength - 645.0) / (645.0 - 580.0)
        Blue = 0.0
    elif 645 <= trunc(Wavelength) and trunc(Wavelength) <= 780:
        Red = 1.0
        Green = 0.0
        Blue = 0.0
    else:
        Red = 0.0
        Green = 0.0
        Blue = 0.0
    # Let the intensity fall off near the vision limits
    if 380 <= trunc(Wavelength) and trunc(Wavelength) <= 419:
        factor = 0.3 + 0.7*(Wavelength - 380.0) / (420.0 - 380.0)
    elif 420 <= trunc(Wavelength) and trunc(Wavelength) <= 700:
        factor = 1.0
    elif 701 <= trunc(Wavelength) and trunc(Wavelength) <= 780:
        factor = 0.3 + 0.7*(780.0 - Wavelength) / (780.0 - 700.0)
    else:
        factor = 0.0
    R = Adjust(Red, factor)
    G = Adjust(Green, factor)
    B = Adjust(Blue, factor)
    return [int(R), int(G), int(B)]
