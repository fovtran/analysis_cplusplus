// euler flip

#include <cmath>
#include <complex>

bool euler_flip(bool value)
{
    return std::pow
    (
        std::complex<float>(std::exp(1.0)), 
        std::complex<float>(0, 1) 
        * std::complex<float>(std::atan(1.0)
        *(1 << (value + 2)))
    ).real() < 0;
}