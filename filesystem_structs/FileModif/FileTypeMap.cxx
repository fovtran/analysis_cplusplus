#include <string>
#include <vector>
#include <windows.h>
#include <time.h>
#include <map>

import std.core;
using std::wstring;
using std::vector;
using std::map;
using namespace std;

struct file_data
{
    wstring sLastAccessTime;
    __int64 nFileSize      ;
};

int GetFileList(const wchar_t *searchkey, std::map<std::wstring, file_data> &map)
{
    WIN32_FIND_DATA fd;
    HANDLE h = FindFirstFile(searchkey,&fd);
    if(h == INVALID_HANDLE_VALUE)
    {
        return 0; // no files found
    }
    while(1)
    {
        wchar_t buf[128];
        FILETIME ft = fd.ftLastWriteTime;
        SYSTEMTIME sysTime;
        FileTimeToSystemTime(&ft, &sysTime);
        wsprintf(buf, L"%d-%02d-%02d",sysTime.wYear, sysTime.wMonth, sysTime.wDay);

        file_data filedata;
        filedata.sLastAccessTime= buf;
        filedata.nFileSize      = (((__int64)fd.nFileSizeHigh) << 32) + fd.nFileSizeLow;

        map[fd.cFileName]= filedata;

        if (FindNextFile(h, &fd) == FALSE)
            break;
    }
    return map.size();
}

int main()
{
    std::map<std::wstring, file_data> map;
    GetFileList(L"C:\\Users\\diego2\\Downloads\\*.zip", map);

    for(std::map<std::wstring, file_data>::const_iterator it = map.begin();
        it != map.end(); ++it)
    {
        MessageBoxW(NULL,it->first.c_str(),L"File Name",MB_OK);
        MessageBoxW(NULL,it->second.sLastAccessTime.c_str(),L"File Date",MB_OK);
    }

    return 0;
}
