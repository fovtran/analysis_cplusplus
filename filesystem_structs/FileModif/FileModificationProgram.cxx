#include <string>
#include <vector>
#include <windows.h>
#include <time.h>
#include <map>
#include "FileModification.h"

import std.core;
using std::wstring;
using std::vector;
using std::map;
using namespace std;

int main()
{
	LPCWSTR fin = L"FileModification.obj";
	FileActionInfo *v = new FileActionInfo(fin, 'C', FILE_ACTION_MOVED);
	FileActionInfo *v1;
	FileActionQueue *q = new FileActionQueue();
	//q->Add(v);
	CHAR *cd= new CHAR[2] {'C','D'};
	PCHAR cp= cd;

	while(true)
	{
		v1 = q->Search(fin, FILE_ACTION_MOVED, cp);
		if( v1 == NULL )
		{
			printf(".");
		}
		else { printf("file moved: %i\n", v1->drive); }
	}

    return 0;
}
