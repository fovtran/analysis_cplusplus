// g++ -O3 -msse2 asmtest1.cpp &&./a.exe

#include <stdio.h>
#include <intrin.h>
#include <iostream>
#include <inttypes.h>

#define u32 uint32_t
#define u64 uint64_t


int main()
{
uint32_t eax, edx;
u32 xgetbv;
u32 index=1;

	__asm__ volatile (".byte 0x0f,0x01,0xd0"
			: "=a" (eax), "=d" (edx)
			: "c" (index)) ;
	xgetbv =  eax + ((u64)edx <<32);

	printf("The code %i gives %i\n", xgetbv);

  return 0;
}
