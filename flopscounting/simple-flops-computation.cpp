#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

uint64_t builtin_popcnt(const uint64_t* buf, size_t len){
  uint64_t cnt = 0;
  for(size_t i = 0; i < len; ++i){
    cnt += __builtin_popcountll(buf[i]);
  }
  return cnt;
}

int main(int argc, char** argv){
  if(argc != 2){
    printf("Usage: %s <buffer size in MB>\n", argv[0]);
    return -1;
  }
  uint64_t size = atol(argv[1]) << 20;
  uint64_t* buffer = (uint64_t*)malloc((size/8)*sizeof(*buffer));

  // Spoil copy-on-write memory allocation on *nix
  for (size_t i = 0; i < (size / 8); i++) {
    buffer[i] = random();#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

uint64_t builtin_popcnt(const uint64_t* buf, size_t len){
  uint64_t cnt = 0;
  for(size_t i = 0; i < len; ++i){
    cnt += __builtin_popcountll(buf[i]);
  }
  return cnt;
}

int main(int argc, char** argv){
  if(argc != 2){
    printf("Usage: %s <buffer size in MB>\n", argv[0]);
    return -1;
  }
  uint64_t size = atol(argv[1]) << 20;
  uint64_t* buffer = (uint64_t*)malloc((size/8)*sizeof(*buffer));

  // Spoil copy-on-write memory allocation on *nix
  for (size_t i = 0; i < (size / 8); i++) {
    buffer[i] = random();
  }
  uint64_t count = 0;
  clock_t tic = clock();
  for(size_t i = 0; i < 10000; ++i){
    count += builtin_popcnt(buffer, size/8);
  }
  clock_t toc = clock();
  printf("Count: %lu\tElapsed: %f seconds\tSpeed: %f GB/s\n", count, (double)(toc - tic) / CLOCKS_PER_SEC, ((10000.0*size)/(((double)(toc - tic)*1e+9) / CLOCKS_PER_SEC)));
  return 0;
}
  }
  uint64_t count = 0;
  clock_t tic = clock();
  for(size_t i = 0; i < 10000; ++i){
    count += builtin_popcnt(buffer, size/8);
  }
  clock_t toc = clock();
  printf("Count: %lu\tElapsed: %f seconds\tSpeed: %f GB/s\n", count, (double)(toc - tic) / CLOCKS_PER_SEC, ((10000.0*size)/(((double)(toc - tic)*1e+9) / CLOCKS_PER_SEC)));
  return 0;
}