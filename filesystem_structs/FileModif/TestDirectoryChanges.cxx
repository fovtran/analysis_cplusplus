#include <iostream>
#include <windows.h>

DWORD processDirectoryChanges(const char *buffer)
{
    DWORD offset = 0;
    char fileName[MAX_PATH] = "";
    FILE_NOTIFY_INFORMATION *fni = NULL;

    do
    {
        fni = (FILE_NOTIFY_INFORMATION*)(&buffer[offset]);
        // since we do not use UNICODE,
        // we must convert fni->FileName from UNICODE to multibyte
        int ret = ::WideCharToMultiByte(CP_ACP, 0, fni->FileName,
            fni->FileNameLength / sizeof(WCHAR),
            fileName, sizeof(fileName), NULL, NULL);

        switch (fni->Action)
        {
        case FILE_ACTION_ADDED:
        {
            std::cout << fileName << std::endl;
        }
        break;
        default:
            break;
        }

        ::memset(fileName, '\0', sizeof(fileName));
        offset += fni->NextEntryOffset;

    } while (fni->NextEntryOffset != 0);

    return 0;
}

int main()
{
    HANDLE hDir = ::CreateFile("C:\\Users\\nenad.smiljkovic\\Desktop\\test",
        FILE_LIST_DIRECTORY,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL, OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, NULL);

    if (INVALID_HANDLE_VALUE == hDir) return ::GetLastError();

    OVERLAPPED ovl = { 0 };
    ovl.hEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL);

    if (NULL == ovl.hEvent) return ::GetLastError();

    DWORD error = 0, br;
    char buffer[1024];

    while (1)
    {
        error = ::ReadDirectoryChangesW(hDir,
            buffer, sizeof(buffer), FALSE,
            FILE_NOTIFY_CHANGE_FILE_NAME,
            NULL, &ovl, NULL);

        if (0 == error)
        {
            error = ::GetLastError();

            if (ERROR_IO_PENDING != ::GetLastError())
            {
                ::CloseHandle(ovl.hEvent);
                ::CloseHandle(hDir);
                return error;
            }
        }

        error = ::WaitForSingleObject(ovl.hEvent, 0);

        switch (error)
        {
        case WAIT_TIMEOUT:
            break;
        case WAIT_OBJECT_0:
        {
            error = processDirectoryChanges(buffer);

            if (error > 0)
            {
                ::CloseHandle(ovl.hEvent);
                ::CloseHandle(hDir);
                return error;
            }

            if (0 == ::ResetEvent(ovl.hEvent))
            {
                error = ::GetLastError();
                ::CloseHandle(ovl.hEvent);
                ::CloseHandle(hDir);
                return error;
            }
        }
        break;
        default:
            error = ::GetLastError();
            ::CloseHandle(ovl.hEvent);
            ::CloseHandle(hDir);
            return error;
            break;
        }
    }

    return 0;
}
