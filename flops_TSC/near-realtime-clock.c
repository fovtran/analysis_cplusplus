#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NANO 1000000000L

// buf needs to store 30 characters
int timespec2str(char *buf, uint len, struct timespec *ts) {
    int ret;
    struct tm t;

    tzset();
    if (localtime_r(&(ts->tv_sec), &t) == NULL)
        return 1;

    ret = strftime(buf, len, "%F %T", &t);
    if (ret == 0)
        return 2;
    len -= ret - 1;

    ret = snprintf(&buf[strlen(buf)], len, ".%09ld", ts->tv_nsec);
    if (ret >= len)
        return 3;

    return 0;
}

int main(int argc, char **argv) {
    clockid_t clk_id = CLOCK_REALTIME;
    const uint TIME_FMT = strlen("2012-12-31 12:59:59.123456789") + 1;
    char timestr[TIME_FMT];

    struct timespec ts, res;
    clock_getres(clk_id, &res);
    clock_gettime(clk_id, &ts);

    if (timespec2str(timestr, sizeof(timestr), &ts) != 0) {
        printf("timespec2str failed!\n");
        return EXIT_FAILURE;
    } else {
        unsigned long resol = res.tv_sec * NANO + res.tv_nsec;
        printf("CLOCK_REALTIME: res=%ld ns, time=%s\n", resol, timestr);
        return EXIT_SUCCESS;
    }
}
