#include<stdio.h>
#include<string.h>
#include<pthread.h>
#include<stdlib.h>
#include<unistd.h>
#include<semaphore.h>

sem_t sema_obj;

pthread_t tid[2];

int shared_val = 0;

void* doSomeThing(void *arg)
{
    unsigned long i = 0;
    pthread_t id = pthread_self();

    for(i=0;i<5;i++){
    printf("\n going to wait %x\n",(unsigned int)id);
    sem_wait(&sema_obj);
    shared_val++;
    sleep(1);
    printf("\n %d The value of shared_val is %d in thread %x \n",(int)i, shared_val, (unsigned int)id);
    sem_post(&sema_obj);
    printf("\n gave up sem %x\n",(unsigned int)id);
    }

    for(i=0; i<(0xFFFFFFFF);i++);

    return NULL;
}

int main(void)
{
    int i = 0;
    int err;
    sem_init(&sema_obj, 0, 1);
    while(i < 2)
    {
    pthread_attr_t attr;
    struct sched_param param;

    pthread_attr_init(&attr);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
    param.sched_priority = 50;
    pthread_attr_setschedparam(&attr, &param);
    //sched_setscheduler(current, SCHED_FIFO, &param);
        err = pthread_create(&(tid[i]), &attr, &doSomeThing, NULL);
    //err = pthread_create(&(tid[i]), NULL, &doSomeThing, NULL);
        if (err != 0)
            printf("\ncan't create thread :[%s]", strerror(err));
        else
            printf("\n Thread created successfully 0x%X \n",(unsigned int)tid[i]);

        i++;
    }
    enter code here
    sleep(60);
    return 0;
}