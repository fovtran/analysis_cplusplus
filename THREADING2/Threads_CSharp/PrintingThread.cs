using System;
using System.Threading;

namespace MultiThreadIII
{
    public class PrintingThread
    {
        public void threadJob()
        {
            Console.WriteLine(Thread.CurrentThread.Name);
            Console.WriteLine("Here are the numbers for: {0}",
                    Thread.CurrentThread.Name);
            for (int i = 0; i < 10; i++)
            {
                Random r = new Random();
                Thread.Sleep(1000 * r.Next(2));
                Console.Write("{0} ", i);
            }
            Console.WriteLine();
        }

        // 10 Thread objects.
        // each object call the same instance of the PrintingThread object.
        public static void Main()
        {
            // make instance
            PrintingThread pt = new PrintingThread();

            // 10 threads that are all pointing to the same method on the same object
            Thread[] ts = new Thread[10];
            for (int i = 0; i < 10; i++)
            {
                ts[i] = new Thread(new ThreadStart(pt.threadJob));
                ts[i].Name = string.Format("Worker thread [{0}]", i);
            }

            foreach (Thread t in ts) t.Start();

            Console.ReadLine();
        }
    }
}
