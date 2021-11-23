-- "c:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

## Bitshifting in MSVC

unsigned char A = 0xB9;
unsigned char B = 0x91;
unsigned char C = A << 3; // shift bits in A three bits to the left.
unsigned char D = B >> 2; // shift bits in B two bits to the right.

AARRGGBB

red = color & 0xff;
green = (color >> 8) & 0xff;
blue = (color >> 16> & 0xff;
alpha = (color >> 24) & 0xff;
Conversely, we can put components together:

color = (alpha << 24) | (blue << 16) | (green << 8) | red;
