char* pipeName =” \\\\.\\pipe\\”;
char* eventToChildName = “event_to_child”;
char* eventFromChildName = “event_from_child”;

//Create the name pipe by the pipe name;
Handle namedPipe = CreateNamedPipe(pipeName,
			PIPE_ACCESS_DUPLEX +FILE_FLAG_FIRST_PIPE_INSTANCE,
			PIPE_TYPE_MESSAGE + PIPE_WAIT + PIPE_READMODE_MESSAGE,
			PIPE_UNLIMITED_INSTANCES,
			100,
			100,
			100,
			nullptr);

//Create the event to child, where eventSecurity is the pointer to
//SECURITY_ATTRIBUTES structure.
Handle eventToChild = CreateEvent(&eventSecurity,
				false,
				false,
				eventToChildName );
//Create the event from child, the child process uses it to notify the parent process.
Handle eventFromChild = CreateEvent(&eventSecurity,false, false, eventFromChildName );

//notify the child process
if (!SetEvent(eventToChild))
	return;

//Write the data to the named pipe
DWORD writtenSize;
if (!WriteFile(namedPipe, data, sizeof(data), & writtenSize, nullptr) || writtenSize!= sizeof(data))
	return;
Now, let’s turn to the child process. The child process also creates the named pipe handle, say “hPipe”, and opens “event to child” and “event from child” based on the same event names. After waiting for “event to child” to be signalled by the parent process, the child process reads the data from the named pipe file:

//waiting for the event sent from the parent process
DWORD wait = WaitForSingleObject( eventToChild, INFINITE );
if(wait != WAIT_OBJECT_0 )
{
	//handling error code
}

//continuously reading the data from the named pipe
bool res = false;
while (1)
	{
	res = ReadFile( hPipe, lpBuffer, nNumberOfBytesToRead, & lpNumberOfBytesRead, nullptr) ;
	if( !res )
		break;
	}
2. To share big data: named shared-memory
After reading the data from the name pipe, let’s assume that the child process generates a big block of data to share with the parent process. The fastest way to handle this is to create a file mapping object, map the file object into memory space, notify the parent process of the file mapping handle and data size, and then dump the data to the mapped buffer for the parent process to read. The parent process locates the memory section using the mapped file handle and the buffer size, which are written in the named pipe, to read the data in the mapped buffer directly.

Here is the code for the child process:

Handle hMapFile = CreateFileMapping(
				INVALID_HANDLE_VALUE,
				NULL,
				PAGE_READWRITE,
				0,
				bufferSize,
				nullptr);

	if ( hMapFile == nullptr)
		return;

	// map the file into memory
	LPSTR mappedBuffer = (LPSTR) MapViewOfFile(
					hMapFile,
					FILE_MAP_ALL_ACCESS,
					0,
					0,
					0 );

//notify the parent process to receive the mapped file handle and the buffer size
(!SetEvent( eventFromChild))
	return;

//After notifying, write the data to the named pipe
DWORD buffer[2];
buffer[0] = (DWORD)hMapFile;
buffer[1] = bufferSize;
if (WriteFile(hPipe, buffer, sizeof buffer, &written, NULL ) || ( written != sizeof buffer ) ))
	return;

//here we can wait for the parent process to return a message that
//it has received the data and is ready to read the buffer data.

//dump the data
memcpy(destination, mappedBuffer, bufferSize);
…
On the other hand, after receiving the mapped file handle “childhMapFile” and the buffer size from the child process, the parent process can use the “DuplicateHandle()” function to duplicate the handle “hMapFile”, and then map the file into current process memory. Here is the code:

if ( DuplicateHandle( childProcess, childhMapFile, currentProcess, & hMapFile, 0, false, DUPLICATE_SAME_ACCESS ) )

	// map the file into our process memory
	LPSTR hMappedBuffer = (LPSTR) MapViewOfFile(
				hMapFile,
				FILE_MAP_ALL_ACCESS,
				0,
				0,
				0);
}
3. Conclusion
Shared data is one of several ways for processes to communicate. Named pipe and shared memory are used in different circumstances. The two processes access the same named pipe by name, and access the shared memory by the mapped file handle. Proper use of event objects and wait functions to control the timing of shared data reading and writing will ensure process synchronization.
