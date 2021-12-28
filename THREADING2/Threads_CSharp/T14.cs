using System;
using System.Threading;
using System.Collections.Generic;

namespace T14
{
    class BasicWaitHandle
    {
        static EventWaitHandle wh = new AutoResetEvent (true);

        public static void T1(  )
        {
            new Thread (Waiter).Start(  );
            Thread.Sleep (1);                  // Pause for a second...
            wh.Set(  );                             // Wake up the Waiter.
        }

        public static void Waiter(  )
        {
            Console.WriteLine ("Waiting...");
            wh.WaitOne(  );                        // Wait for notification
            Console.WriteLine ("Notified");
        }
        }
}