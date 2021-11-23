#include <stdio.h>
#include <stdlib.h>
#include <editline/readline.h>
#include <editline/history.h>

int main(int argc, char** argv) {

/* version/exit info */
puts("Edward Version 0.0.1");
puts("Press Ctrl+c to Exit\n");

/* endless loop for main REPL */
while (1) {
 	/* output prompt and read line */
 	char* input = readline("lispy> ");
 	/* put input in history  */
 	add_history(input);
 	/* Echo input back */
 	printf("No you're a %s\n", input);
 	/* free input */
 	free(input);
 }
 return 0;
 } 
