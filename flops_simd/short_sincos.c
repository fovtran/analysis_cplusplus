Over 100000000 test, milianw answer is 2 time slower than std::cos implementation.
However, you can manage to run it faster by doing the following steps:

->use float
->don't use floor but static_cast
->don't use abs but ternary conditional
->use #define constant for division
->use macro to avoid function call

// 1 / (2 * PI)
#define FPII 0.159154943091895
//PI / 2
#define PI2 1.570796326794896619

#define _cos(x)         x *= FPII;\
                        x -= .25f + static_cast<int>(x + .25f) - 1;\
                        x *= 16.f * ((x >= 0 ? x : -x) - .5f);
#define _sin(x)         x -= PI2; _cos(x);
