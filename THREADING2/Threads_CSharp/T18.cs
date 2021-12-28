using System;
using System.Threading;
using System.Collections.Generic;

namespace T18
{
    public class TaskQueue : IDisposable
    {
    object locker = new object(  );
    Thread[] workers;
    Queue<string> taskQ = new Queue<string>(  );

    public TaskQueue (int workerCount)
    {
        workers = new Thread [workerCount];

        // Create and start a separate thread for each worker
        for (int i = 0; i < workerCount; i++)
        (workers [i] = new Thread (Consume)).Start(  );
    }

    public void Dispose(  )
    {
        // Enqueue one null task per worker to make each exit.
        foreach (Thread worker in workers) EnqueueTask (null);
    }

    public void EnqueueTask (string task)
    {
        lock (locker)
        {
        taskQ.Enqueue (task);            // We must pulse because we're
        Monitor.Pulse (locker);          // changing a blocking condition.
        }
    }

    void Consume(  )
    {
        while (true)                        // Keep consuming until
        {                                   // told otherwise
        string task;
        lock (locker)
        {
            while (taskQ.Count == 0) Monitor.Wait (locker);
            task = taskQ.Dequeue(  );
        }
        if (task == null) return;         // This signals our exit
        Console.Write (task);             // Perform task.
        Thread.Sleep (1000);              // Simulate time-consuming task
        }
    }

        public static void T1(  )
        {
            using (TaskQueue q = new TaskQueue (20))
            {
                for (int i = 0; i < 1000; i++)
                q.EnqueueTask (" Task" + i);

                Console.WriteLine ("Enqueued 10 tasks");
                Console.WriteLine ("Waiting for tasks to complete...");
            }

            // Exiting the using statement runs TaskQueue's Dispose method, which
            // shuts down the consumers, after all outstanding tasks are completed.

            Console.WriteLine ("\r\nAll tasks done!");
            }
    }
}