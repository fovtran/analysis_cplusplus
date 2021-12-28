#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

struct buffer{
    int a;
    char string[10];
};

void* thread1_function(void *ptr){
    struct buffer *buffer=(struct buffer*)ptr;
    printf("hello world\n");
    printf("%s-%d\n", buffer->string,buffer->a);
    return NULL;
}

int main(){

    int err;
    pthread_t thread1;
    struct buffer *buffer;

    buffer = (struct buffer*)malloc(sizeof (struct buffer) );
    buffer->a=1;
    snprintf(buffer->string, sizeof buffer->string, "%s", "strint");
    printf("main: %s - %d\n", buffer->string, buffer->a);

    err = pthread_create(&thread1, NULL, &thread1_function, buffer);
    printf("error: %d\n", err);
    pthread_join(thread1, NULL);

    return 0;
}