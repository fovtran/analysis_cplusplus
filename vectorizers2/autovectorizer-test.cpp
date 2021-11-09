// cl.exe /O2 /fp:fast /arch:AVX2
#include <windows.h>
#include <iostream>
#include <process.h>
#include <stdlib.h>

using namespace std;

float sum = 0.0;
static const int length = 1024 * 8;

static float a[length];
float scalarAverage() {
		for (uint32_t j = 0; j < _countof(a); ++j) {
				sum += a[j];
		}
		 return sum / _countof(a);
	}

unsigned int __stdcall mythreadA(void* data)
{
	for (uint32_t j = 0; j < _countof(a); ++j)
	{
		
	}
		cout << scalarAverage() << endl;
		return 0;
}

int main(int argc, char* argv[]){
	HANDLE myhandleA;

	myhandleA = (HANDLE)_beginthreadex(0,0, &mythreadA, 0, 0,0);
	WaitForSingleObject(myhandleA, INFINITE);
	CloseHandle(myhandleA);
	return EXIT_SUCCESS;
}
