using System;
using System.Text;
using System.IO;
using System.Threading;

namespace Threads
{
    public class App1
    {
        private static void test1()
        {
            T18.TaskQueue.T1();
        }

        public static void Main(string[] args)
        {
            App1.test1();
            Environment.Exit(0);
        }
    }
}