You'll want the std::sin() function from <cmath>.

long double sine_table[2001];
for (int index = 0; index < 2001; index++)
{
    sine_table[index] = std::sin(PI * (index - 1000) / 1000.0);
}


double table[1000] = {0};
for (int i = 1; i <= 1000; i++)
{
    sine_table[i-1] = std::sin(PI * i/ 1000.0);
}


double getSineValue(int multipleOfPi){
     if(multipleOfPi == 0) return 0.0;
     int sign = 1;
     if(multipleOfPi < 0){
         sign = -1;

     }
     return signsine_table[signmultipleOfPi - 1];
}

#another sine approximation
streamin ramp;
streamout sine;

float x,rect,k,i,j;

x = ramp -0.5;

rect = x * (1 - x < 0 & 2);
k = (rect + 0.42493299) *(rect -0.5) * (rect - 0.92493302) ;
i = 0.436501 + (rect * (rect + 1.05802));
j = 1.21551 + (rect * (rect - 2.0580201));
sine = i*j*k*60.252201*x;