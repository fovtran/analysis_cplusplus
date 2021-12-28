/* convert raw 4-byte float data to numerical */

#include <stdio.h>

FILE *input_file, *output_file;
int i;
float j;

int main(int argc, char *argv[]) {

if ( argc < 5 ) {
   printf("\nUsage: raw2num <input file> <output file> <offset> <number of numbers>\n");
   printf("Exampel: ./raw2num adc_data adc_num 65536 32768\n");
   exit(1);
   }

if ( (input_file = fopen(argv[1],"r")) == NULL ) {
        perror("Input file");
        exit(1);
        }
if ( (output_file = fopen(argv[2],"w")) == NULL ) {
        perror("Output file");
        exit(1);
        }


fseek(input_file,atoi(argv[3]),SEEK_SET);

for (i=0;i<atoi(argv[4]);i++) {
    if (fread(&j,4,1,input_file) == 0) {
	if (feof(input_file))
	    printf("Input reached EOF\n");
	else
	    printf("Error with input file\n");
	break;
	}
    fprintf(output_file,"%d %f\n",i,j);
    }

fclose(input_file);
fclose(output_file);
}



