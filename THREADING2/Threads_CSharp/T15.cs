using System;
using System.Threading;
using System.Collections.Generic;

namespace T15
{
    class TwoWaySignaling
    {
    static EventWaitHandle ready = new AutoResetEvent (false);
    static EventWaitHandle go = new AutoResetEvent (false);
    static volatile string message;         // We must either use volatile
                                            // or lock around this field
    public static void T1(  )
    {
        new Thread (Work).Start(  );

        ready.WaitOne(  );            // First wait until worker is ready
        message = "ooo";
        go.Set(  );                   // Tell worker to go!

        ready.WaitOne(  );
        message = "ahhh";           // Give the worker another message
        go.Set(  );

        ready.WaitOne(  );
        message = null;             // Signal the worker to exit
        go.Set(  );
    }

    static void Work(  )
    {
        while (true)
        {
        ready.Set(  );                          // Indicate that we're ready
        go.WaitOne(  );                         // Wait to be kicked off...
        if (message == null) return;          // Gracefully exit
        Console.WriteLine (message);
        }
    }
    } 
}