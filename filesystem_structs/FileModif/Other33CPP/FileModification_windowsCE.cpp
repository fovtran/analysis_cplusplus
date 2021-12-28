void FileInfoHelper::WatchFileChanges( TCHAR *ptcFileBaseDir, TCHAR *ptcFileName ){
static int iCount = 0;
DWORD dwWaitStatus;
HANDLE dwChangeHandles;

if( ! ptcFileBaseDir || ! ptcFileName ) return;

wstring wszFileNameToWatch = ptcFileName;

dwChangeHandles = FindFirstChangeNotification(
    ptcFileBaseDir,
    FALSE,
    FILE_NOTIFY_CHANGE_FILE_NAME |
    FILE_NOTIFY_CHANGE_DIR_NAME |
    FILE_NOTIFY_CHANGE_ATTRIBUTES |
    FILE_NOTIFY_CHANGE_SIZE |
    FILE_NOTIFY_CHANGE_LAST_WRITE |
    FILE_NOTIFY_CHANGE_LAST_ACCESS |
    FILE_NOTIFY_CHANGE_CREATION |
    FILE_NOTIFY_CHANGE_SECURITY |
    FILE_NOTIFY_CHANGE_CEGETINFO
    );

if (dwChangeHandles == INVALID_HANDLE_VALUE)
{
    printf("\n ERROR: FindFirstChangeNotification function failed [%d].\n", GetLastError());
    return;
}

while (TRUE)
{
    // Wait for notification.
    printf("\n\n[%d] Waiting for notification...\n", iCount);
    iCount++;

    dwWaitStatus = WaitForSingleObject(dwChangeHandles, INFINITE);
    switch (dwWaitStatus)
    {
        case WAIT_OBJECT_0:

            printf( "Change detected\n" );

            DWORD iBytesReturned, iBytesAvaible;
            if( CeGetFileNotificationInfo( dwChangeHandles, 0, NULL, 0, &iBytesReturned, &iBytesAvaible) != 0 )
            {
                std::vector< BYTE > vecBuffer( iBytesAvaible );

                if( CeGetFileNotificationInfo( dwChangeHandles, 0, &vecBuffer.front(), vecBuffer.size(), &iBytesReturned, &iBytesAvaible) != 0 ) {
                    BYTE* p_bCurrent = &vecBuffer.front();
                    PFILE_NOTIFY_INFORMATION info = NULL;

                    do {
                        info = reinterpret_cast<PFILE_NOTIFY_INFORMATION>( p_bCurrent );
                        p_bCurrent += info->NextEntryOffset;

                        if( wszFileNameToWatch.compare( info->FileName ) == 0 )
                        {
                            wcout << "\n\t[" << info->FileName << "]: 0x" << ::hex << info->Action;

                            switch(info->Action) {
                                case FILE_ACTION_ADDED:
                                    break;
                                case FILE_ACTION_MODIFIED:
                                    break;
                                case FILE_ACTION_REMOVED:
                                    break;
                                case FILE_ACTION_RENAMED_NEW_NAME:
                                    break;
                                case FILE_ACTION_RENAMED_OLD_NAME:
                                    break;
                            }
                        }
                    }while (info->NextEntryOffset != 0);
                }
            }

            if ( FindNextChangeNotification( dwChangeHandles ) == FALSE )
            {
                printf("\n ERROR: FindNextChangeNotification function failed [%d].\n", GetLastError());
                return;
            }

            break;

        case WAIT_TIMEOUT:
            printf("\nNo changes in the timeout period.\n");
            break;

        default:
            printf("\n ERROR: Unhandled dwWaitStatus [%d].\n", GetLastError());
            return;
            break;
    }
}

FindCloseChangeNotification( dwChangeHandles );
}
