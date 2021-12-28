int foo() {
  int a;             /* integer */
  int *b;            /* pointer to integer */
  char c[10];        /* character arrays */
  char d[3];

  b = &a;            /* initialize b to point to location of a */
  strcpy(c,get_c()); /* get c from somewhere, write it to c */
  *b = 5;            /* the data at the point in memory b indicates is set to 5 */
  strcpy(d,get_d());
  return *b;         /* read from b and pass it to the caller */
}
