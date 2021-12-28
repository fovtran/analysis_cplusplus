using System;
using System.Text;
using System.IO;
using System.Threading;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Linq;

namespace ThreadManager
{    
    class SmartObject
    {
        DateTime ultimo = DateTime.Now;
        public SmartObject(){}
    }

    public class App1
    {
        [DllImport("kernel32")]
        static extern int GetCurrentThreadId();
        [DllImport("kernel32.dll")]
        public static extern int GetCurrentProcessorNumber();

        static bool done;
        static readonly object locker = new object();
        volatile int threadStarter = 0;

        private static void Thread1()
        {
            lock(locker)
            {
                if (!done)
                {
                    // Here goes thread 1
                }
            }
        }

        private static void Thread2()
        {
            lock(locker)
            {
                if (!done)
                {
                    // Here goes thread 2
                }
            }
        }

        public static void Thread3()
        {
            lock(locker)
            {
                int timedivisor = 1000000;
                DateTime ultimo = DateTime.Now;
                if (!done)
                {
                    int i = 0;                    
                    //Thread.BeginThreadAffinity();
                    //CurrentThread.ProcessorAffinity = new IntPtr(1);
                    while (true)
                            {
                                i++;
                                if (i == int.MaxValue)
                                {
                                    i = 0;
                                    var lps = int.MaxValue / (DateTime.Now - ultimo).TotalSeconds / timedivisor;
                                    Console.WriteLine(lps.ToString("0.000") + " M loops/s");
                                    ultimo = DateTime.Now;
                                }
                            }
                }
            }
        }

        private static ProcessThread CurrentThread
        {
            get
            {
                int id = GetCurrentThreadId();
                return Process.GetCurrentProcess().Threads.Cast<ProcessThread>().Single(x => x.Id == id);
            }
        }

        [STAThread]
        public static void Main(string[] args)
        {
        	//Thread a = new Thread(Thread1);
            //Thread b = new Thread(Thread2);
            Thread c = new Thread(Thread3);
            c.Start();
            c.Join();
            //Thread d = new Thread(()=> {});

            Environment.Exit(0);
        }
    }
}