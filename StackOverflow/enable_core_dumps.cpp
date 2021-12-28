#include <sys/resource.h>

bool enable_core_dump(){
    struct rlimit corelim;

    corelim.rlim_cur = RLIM_INFINITY;
    corelim.rlim_max = RLIM_INFINITY;

    return (0 == setrlimit(RLIMIT_CORE, &corelim));
}
