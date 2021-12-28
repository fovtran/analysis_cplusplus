using System;
using System.Threading;

namespace T6
{
    class test1
    {
        public static void T2()
        {
            WorkInvoker method = Work;
            method.BeginInvoke ("test", Done, method);
            // ...
            //
        }

        delegate int WorkInvoker (string text);
        static int Work (string s) { return s.Length; }

        static void Done (IAsyncResult cookie)
        {
            WorkInvoker method = (WorkInvoker) cookie.AsyncState;
            int result = method.EndInvoke (cookie);
            Console.WriteLine ("String length is: " + result);
        }
}
}