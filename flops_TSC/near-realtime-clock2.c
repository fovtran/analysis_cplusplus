#include <stdio.h>
#include <time.h>   /* Needed for struct timespec */

void nsleep2(long miliseconds) {
	struct timespec delta = {5 /*secs*/, 135 /*nanosecs*/};
	while (nanosleep(&delta, &delta))
		printf(".");
}

int nsleep(long miliseconds) {
	struct timespec req, rem;

   if(miliseconds > 999)
   {
        req.tv_sec = (int)(miliseconds / 1000);                            /* Must be Non-Negative */
        req.tv_nsec = (miliseconds - ((long)req.tv_sec * 1000)) * 1000000; /* Must be in range of 0 to 999999999 */
   }
   else
   {
        req.tv_sec = 0;                         /* Must be Non-Negative */
        req.tv_nsec = miliseconds * 1000000;    /* Must be in range of 0 to 999999999 */
   }

   return nanosleep(&req , &rem);
}

int main() {
	int ret = nsleep(2500);
	printf("sleep result %d\n",ret);
	nsleep2(150);
   return 0;
}
