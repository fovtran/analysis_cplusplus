#include <stdio.h>
#include <intrin.h>
#pragma intrinsic(__rdtsc)

  // time_in_seconds = number_of_clock_cycles / frequency
int main()
{
	unsigned __int64 t;
	t = __rdtsc();
	// Stuff
	t = __rdtsc() - t;
    printf_s("%I64d ticks\n", t);
}
