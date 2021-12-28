#include <stdio.h>

void genimage(unsigned char *image, char* filname);

int main(){
	unsigned char image;
	char* filname = "image.bmp";
	genimage((unsigned char*) image, filname);
	printf("File created\n");
}


void genimage(unsigned char* image, char* filname)
{
	unsigned char padding[3] = {0,0,0};
	
	FILE* imgfile = fopen(filname, "wb");
	fwrite(padding, 1, sizeof(padding), imgfile);
	fclose(imgfile);
	//free(padding);
}
