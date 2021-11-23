#include
void main()
{
	int A[20], N, Temp, i, j;
	clrscr();
	printf("\n\n\t ENTER THE NUMBER OF TERMS...: ");
	scanf("%d", &N);
	printf("\n\t ENTER THE ELEMENTS OF THE ARRAY...:");
	for(i=0; i<N; i++)
	{
		scanf("\n\t\t%d", &A[i]);
	}
	for(i=1; i<N; i++)
	{
		Temp = A[i];
		j = i-1;
		while(Temp<A[j] && j>=0)
		{
			A[j+1] = A[j];
			j = j-1;
		}
		A[j+1] = Temp;
	}
	printf("\n\tTHE ASCENDING ORDER LIST IS...:\n");
	for(i=0; i<N; i++)
		printf("\n\t\t\t%d", A[i]);
	getch();
}