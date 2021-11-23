// The author of this software is Shane O'Sullivan.
#include <stdio.h>
#include <search.h>
#include <malloc.h>
#include "VoronoiDiagramGenerator.h"

int main(int argc,char **argv)
{
	float xValues[4] = {-22, -17, 4,22};
	float yValues[4] = {-9, 31,13,-5};

	long count = 4;

	VoronoiDiagramGenerator vdg;
	vdg.generateVoronoi(xValues,yValues,count, -100,100,-100,100,3);

	vdg.resetIterator();

	float x1,y1,x2,y2;

	printf("\n-------------------------------\n");
	while(vdg.getNext(x1,y1,x2,y2))
	{
		printf("GOT Line (%f,%f)->(%f,%f)\n",x1,y1,x2, y2);
	}
	return 0;
}
