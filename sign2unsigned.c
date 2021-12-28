/* convert raw 2-byte data to numerical */

#include <stdio.h>

FILE *input_file, *output_file;
int i,j;

int main(int argc, char *argv[]) {

if ( argc < 6 ) {
   printf("\nUsage: raw2num <input file> <output file> <file offset> <number of numbers> <bias>\n");
   printf("Exampel: ./raw2num adc_data adc_num 50000 100000\n");
   exit(1);
   }

input_file = fopen(argv[1],"r");
output_file = fopen(argv[2],"w");

fseek(input_file,atoi(argv[3]),SEEK_SET);

for (i=0;i<atoi(argv[4]);i++) {
    fread(&j,2,1,input_file);
    if (j>32767)
       j-=32768;
    else
       j+=32768;
    j+=atoi(argv[5]);
    fwrite(&j,2,1,output_file);
    }

fclose(input_file);
fclose(output_file);
}



