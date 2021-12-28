#include <stdio.h>
#include <stdlib.h>
#include <ftw.h>

char buf[4096];
struct stat st;
int fd;
off_t pos;
ssize_t written;

memset(buf, 0, 4096);
fd = open(file_to_overwrite, O_WRONLY);
fstat(fd, &st);

for (pos = 0; pos < st.st_size; pos += written)
    if ((written = write(fd, buf, min(st.st_size - pos, 4096))) <= 0)
        break;

fsync(fd);
close(fd);
Option two:

int fd = open(file_to_truncate, O_WRONLY);
ftruncate(fd, 0);
fsync(fd);
close(fd);