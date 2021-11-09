# xmm0

m128_f32
m128_f64

m128_i8
m128_i16
m128_i32
m128_i64

m128_u8
m128_u16
m128_u32
m128_u64


float f1 = 1.0f;
  00007FF61DAC185C  vmovss      xmm0,dword ptr [__real@3f800000 (xxxh)]  
> 00007FF61DAC1864  vmovss      dword ptr [f1],xmm0  

complex<float> fc=2.0 + 3.0i;
  00007FF61DAC186A  vmovsd      xmm1,qword ptr [__real@4008000000000000 (xxxh)]  

evals to: 2.452e-43#DEN
