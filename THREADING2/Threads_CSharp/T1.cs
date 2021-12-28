using System;
using System.Threading;

namespace T1
{
class test
{

    public static void T1()
    {
        Thread t = new Thread (WriteY);          // Kick off a new thread
        t.Start();                               // running WriteY(  )

            // Simultaneously, do something on the main thread.
            for (int i = 0; i < 1000; i++) Console.Write ("x");
    }

    static void WriteY(  )
    {
    for (int i = 0; i < 1000; i++) Console.Write ("y");
    }

    public static void T2()
        {
            Thread t = new Thread (Go);
            t.Start(  );
            t.Join(  );
            Console.WriteLine ("Thread t has ended!");
        }

    static void Go(  ) { for (int i = 0; i < 1000; i++) Console.Write ("y"); }

    public static void T3()
    {
        Thread t = new Thread (Print);
        t.Start ("Hello from t!");
        Print ("Hello from the main thread!");
    }

    static void Print (object messageObj)
    {
        string message = (string) messageObj;
        Console.WriteLine (message);
    }

    public void T4()
        {
        string text = "t1";
        Thread t1 = new Thread (delegate(  ) { Print (text); });

        text = "t2";
        Thread t2 = new Thread (delegate(  ) { Print (text); });

        t1.Start(  );
        t2.Start(  );
        }
    static void Print (string message) { Console.WriteLine (message); }


}
}