inline static void alignedMemcpySSE(void *dst, const void * src, size_t length)
{
#if defined(__x86_64__) || defined(__i386__)
  size_t rem = (7 - ((length & 0x7F) >> 4)) * 10;
  void * end = dst + (length & ~0x7F);

  __asm__ __volatile__ (
   // save the registers we intend to alter, failure to do so causes problems
   // when gcc -O3 is used
   "push        %[dst]\n\t"
   "push        %[src]\n\t"
   "push        %[end]\n\t"

   "cmp         %[dst],%[end] \n\t"
   "je          remain_%= \n\t"

   // perform 128 byte SIMD block copy
   "loop_%=: \n\t"
   "vmovaps     0x00(%[src]),%%xmm0 \n\t"
   "vmovaps     0x10(%[src]),%%xmm1 \n\t"
   "vmovaps     0x20(%[src]),%%xmm2 \n\t"
   "vmovaps     0x30(%[src]),%%xmm3 \n\t"
   "vmovaps     0x40(%[src]),%%xmm4 \n\t"
   "vmovaps     0x50(%[src]),%%xmm5 \n\t"
   "vmovaps     0x60(%[src]),%%xmm6 \n\t"
   "vmovaps     0x70(%[src]),%%xmm7 \n\t"
   "vmovntdq    %%xmm0,0x00(%[dst]) \n\t"
   "vmovntdq    %%xmm1,0x10(%[dst]) \n\t"
   "vmovntdq    %%xmm2,0x20(%[dst]) \n\t"
   "vmovntdq    %%xmm3,0x30(%[dst]) \n\t"
   "vmovntdq    %%xmm4,0x40(%[dst]) \n\t"
   "vmovntdq    %%xmm5,0x50(%[dst]) \n\t"
   "vmovntdq    %%xmm6,0x60(%[dst]) \n\t"
   "vmovntdq    %%xmm7,0x70(%[dst]) \n\t"
   "add         $0x80,%[dst] \n\t"
   "add         $0x80,%[src] \n\t"
   "cmp         %[dst],%[end] \n\t"
   "jne         loop_%= \n\t"

   "remain_%=: \n\t"

   // copy any remaining 16 byte blocks
#ifdef __x86_64__
   "leaq        (%%rip), %[end]\n\t"
   "add         $10,%[end] \n\t"
#else
   "call        .+5 \n\t"
   "pop         %[end] \n\t"
   "add         $8,%[end] \n\t"
#endif
   "add         %[rem],%[end] \n\t"
   "jmp         *%[end] \n\t"

   // jump table
   "vmovaps     0x60(%[src]),%%xmm0 \n\t"
   "vmovntdq    %%xmm0,0x60(%[dst]) \n\t"
   "vmovaps     0x50(%[src]),%%xmm1 \n\t"
   "vmovntdq    %%xmm1,0x50(%[dst]) \n\t"
   "vmovaps     0x40(%[src]),%%xmm2 \n\t"
   "vmovntdq    %%xmm2,0x40(%[dst]) \n\t"
   "vmovaps     0x30(%[src]),%%xmm3 \n\t"
   "vmovntdq    %%xmm3,0x30(%[dst]) \n\t"
   "vmovaps     0x20(%[src]),%%xmm4 \n\t"
   "vmovntdq    %%xmm4,0x20(%[dst]) \n\t"
   "vmovaps     0x10(%[src]),%%xmm5 \n\t"
   "vmovntdq    %%xmm5,0x10(%[dst]) \n\t"
   "vmovaps     0x00(%[src]),%%xmm6 \n\t"
   "vmovntdq    %%xmm6,0x00(%[dst]) \n\t"

   // alignment as the previous two instructions are only 4 bytes
   "nop\n\t"
   "nop\n\t"

   // restore the registers
   "pop         %[end]\n\t"
   "pop         %[src]\n\t"
   "pop         %[dst]\n\t"
   :
   : [dst]"r" (dst),
     [src]"r" (src),
     [end]"c" (end),
     [rem]"d" (rem)
   : "xmm0",
     "xmm1",
     "xmm2",
     "xmm3",
     "xmm4",
     "xmm5",
     "xmm6",
     "xmm7",
     "memory"
  );

  //copy any remaining bytes
  for(size_t i = (length & 0xF); i; --i)
    ((uint8_t *)dst)[length - i] =
      ((uint8_t *)src)[length - i];
#else
  memcpy(dst, src, length);
#endif
}