#include <windows.h>
#include <iostream>

DWORD WINAPI myThread(LPVOID lpParameter)
{
	unsigned int& myCounter = *((unsigned int*)lpParameter);
	while(myCounter < 0xFFFFFFFF) ++myCounter;
	return 0;
}

int main(int argc, char* argv[])
{
	using namespace std;

	unsigned int myCounter = 0;
	DWORD myThreadID;
	HANDLE myHandle = CreateThread(0, 0, myThread, &myCounter;, 0, &myThreadID;);
	char myChar = ' ';
	while(myChar != 'q') {
		cout << myCounter << endl;
		myChar = getchar();
	}

	CloseHandle(myHandle);
	return 0;
}
/*
HANDLE WINAPI CreateThread(
__in_opt LPSECURITY_ATTRIBUTES lpThreadAttributes,
__in SIZE_T dwStackSize,
__in LPTHREAD_START_ROUTINE lpStartAddress,
__in_opt LPVOID lpParameter,
__in DWORD dwCreationFlags,
__out_opt LPDWORD lpThreadId
);
Parameters

lpThreadAttributes [in, optional]
A pointer to a SECURITY_ATTRIBUTES structure that determines whether the returned handle can be inherited by child processes. If lpThreadAttributes is NULL, the handle cannot be inherited.
The lpSecurityDescriptor member of the structure specifies a security descriptor for the new thread. If lpThreadAttributes is NULL, the thread gets a default security descriptor. The ACLs in the default security descriptor for a thread come from the primary token of the creator.
dwStackSize [in]
The initial size of the stack, in bytes. The system rounds this value to the nearest page. If this parameter is zero, the new thread uses the default size for the executable.
lpStartAddress [in]
A pointer to the application-defined function to be executed by the thread. This pointer represents the starting address of the thread.
lpParameter [in, optional]
A pointer to a variable to be passed to the thread.
dwCreationFlags [in]
The flags that control the creation of the thread.
lpThreadId [out, optional]
A pointer to a variable that receives the thread identifier. If this parameter is NULL, the thread identifier is not returned.
*/

#include <Windows.h>
#include <stdio.h>

DWORD WINAPI mythread(__in LPVOID lpParameter)
{
	printf("Thread inside %d \n", GetCurrentThreadId());
	return 0;
}

int main(int argc, char* argv[])
{
	HANDLE myhandle;
	DWORD mythreadid;
	myhandle = CreateThread(0, 0, mythread, 0, 0, &mythreadid;);
	printf("Thread after %d \n", mythreadid);
	getchar();
	return 0;
}

/*
uintptr_t _beginthread(
   void( *start_address )( void * ),
   unsigned stack_size,
   void *arglist
);

uintptr_t _beginthreadex(
   void *security,
   unsigned stack_size,
   unsigned ( *start_address )( void * ),
   void *arglist,
   unsigned initflag,
   unsigned *thrdaddr
);

start_address
Start address of a routine that begins execution of a new thread. For _beginthread, the calling convention is either __cdecl or __clrcall; for _beginthreadex, it is either __stdcall or __clrcall.
stack_size
Stack size for a new thread or 0.
arglist
Argument list to be passed to a new thread or NULL.
security
Pointer to a SECURITY_ATTRIBUTES structure that determines whether the returned handle can be inherited by child processes. If NULL, the handle cannot be inherited.
initflag
Initial state of a new thread (0 for running or CREATE_SUSPENDED for suspended); use ResumeThread to execute the thread.
thrdaddr
Points to a 32-bit variable that receives the thread identifier. Might be NULL, in which case it is not used.
*/
