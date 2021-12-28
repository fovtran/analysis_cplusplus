#include <windows.h>
#include <stdio.h>

#define THREADCOUNT 4
#define DLL_NAME TEXT("testdll")

VOID ErrorExit(LPSTR);

extern "C" BOOL WINAPI StoreData(DWORD dw);
extern "C" BOOL WINAPI GetData(DWORD *pdw);

DWORD WINAPI ThreadFunc(VOID)
{
   int i;

   if(!StoreData(GetCurrentThreadId()))
      ErrorExit("StoreData error");

   for(i=0; i<THREADCOUNT; i++)
   {
      DWORD dwOut;
      if(!GetData(&dwOut))
         ErrorExit("GetData error");
      if( dwOut != GetCurrentThreadId())
         printf("thread %d: data is incorrect (%d)\n", GetCurrentThreadId(), dwOut);
      else printf("thread %d: data is correct\n", GetCurrentThreadId());
      Sleep(0);
   }
   return 0;
}

int main(VOID)
{
   DWORD IDThread;
   HANDLE hThread[THREADCOUNT];
   int i;
   HMODULE hm;

// Load the DLL

   hm = LoadLibrary(DLL_NAME);
   if(!hm)
   {
      ErrorExit("DLL failed to load");
   }

// Create multiple threads.

   for (i = 0; i < THREADCOUNT; i++)
   {
      hThread[i] = CreateThread(NULL, // default security attributes
         0,                           // use default stack size
         (LPTHREAD_START_ROUTINE) ThreadFunc, // thread function
         NULL,                    // no thread function argument
         0,                       // use default creation flags
         &IDThread);              // returns thread identifier

   // Check the return value for success.
      if (hThread[i] == NULL)
         ErrorExit("CreateThread error\n");
   }

   WaitForMultipleObjects(THREADCOUNT, hThread, TRUE, INFINITE);

   FreeLibrary(hm);

   return 0;
}

VOID ErrorExit (LPSTR lpszMessage)
{
   fprintf(stderr, "%s\n", lpszMessage);
   ExitProcess(0);
}
