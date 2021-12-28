#include <stdint.h>
#include <stdio.g>
#include <iostream>
#include <filesystem>
// https://channel9.msdn.com/Browse/AllContent
// cd / | find . -name "stdio.h" find . -name "std*.h"
// cl /experimental:module /EHsc /MD /std:c++latest test-vs2017-slm.cxx 

const int bytesPerPixel = 3;
const int fileHeaderSize = 14;
const int infoHeaderSize = 40;

void genimage(unsigned char *image, int h, int w, char* filname);
unsigned char* createHeader(int h, int w, int paddingSize);
unsigned char* createInfoHeader(int h, int w);

int main(){
	int _h= 640;
	int _w= 480;
	unsigned char image[_h][_w][bytesPerPixel];
	unsigned char image;
	char* filname = "image.bmp";
	genimage((unsigned char*) image, _h, _w, filname);
	printf("File created\n");
}

void genimage(unsigned char* image, int h, int w, char* filname)
{
	unsigned char padding[3] = {0,0,0};
	int paddingSize = (4- (w, bytesPerPixel)%4) %4;

	unsigned char* fileHeader = createHeader(h, w, paddingSize);
	unsigned char* infoHeader = createInfoHeader(h, w);

	FILE* imgfile = fopen(filname, "wb");
	fwrite(fileHeader, 1, fileHeaderSize, imgfile);
	fwrite(infoHeader, 1, infoHeaderSize, imgfile);

	fclose(imgfile);
	free(fileHeader);
	free(infoHeader);
}

void createHeader(int h, int w, int paddingSize){
	int fileSize = fileHeaderSize + infoHeaderSize + (bytesPerPixel*w*paddingSize) * h;
	static unsigned char fileHeader[] = {0,0,0,0};
	fileHeader[0] = (unsigned char)('B');
	return fileHeader;

}
void createInfoHeader(int h, int w){
	static unsigned char infoHeader[] = {0,0,0,0};

	infoHeader[0] = (unsigned char) (infoHeaderSize);
	return infoHeader;
}
