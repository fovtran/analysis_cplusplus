using System;
using System.Threading;

namespace T4
{
    class test
    {
        public static void T1()
        {
            for (int i = 0; i < 50; i++) ThreadPool.QueueUserWorkItem (Go);
        }

    static void Go (object notUsed)
    {
        // Compute a hash on a 100,000 byte random byte sequence:
        byte[] data = new byte [100000000];
        new Random(  ).NextBytes (data);
        System.Security.Cryptography.SHA1.Create(  ).ComputeHash (data);
    }
}
}