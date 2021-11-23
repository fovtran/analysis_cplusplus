
This is an implementation of Taylor Series method
unsigned int Math::SIN_LOOP = 15;
unsigned int Math::COS_LOOP = 15;

// sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
template <class T>
T Math::sin(T x)
{
    T Sum       = 0;
    T Power     = x;
    T Sign      = 1;
    const T x2  = x * x;
    T Fact      = 1.0;
    for (unsigned int i=1; i<SIN_LOOP; i+=2)
    {
        Sum     += Sign * Power / Fact;
        Power   *= x2;
        Fact    *= (i + 1) * (i + 2);
        Sign    *= -1.0;
    }
    return Sum;
}

// cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...
template <class T>
T Math::cos(T x)
{
    T Sum       = x;
    T Power     = x;
    T Sign      = 1.0;
    const T x2  = x * x;
    T Fact      = 1.0;
    for (unsigned int i=3; i<COS_LOOP; i+=2)
    {
        Power   *= x2;
        Fact    *= i * (i - 1);
        Sign    *= -1.0;
        Sum     += Sign * Power / Fact;
    }
    return Sum;
}