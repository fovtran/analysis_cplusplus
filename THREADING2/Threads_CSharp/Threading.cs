// Simple Thread Lock Code Example in C#
// Michael Adams, 2-5-2006
// cwolfsheep@yahoo.com

using System.Threading;

// Thread variable
private Thread cmdThread;

// Sample function to implement the wait and queued thread
void startThread()
{
   try{
   waitOnThread(); //force thread wait if other thread is active

   ThreadStart threadstart = new ThreadStart(threadedFunction());
   cmdThread = new Thread(threadstart);
   cmdThread.Start();
   }

   catch
   {
   }
}

// Sample function to load the threaded event
void threadedFunction()
{
   //Put your threaded event content here
}

// Wait on any running thread
void waitOnThread()
{
   try{
   //Update display if needed (indicate wait is going to occur)
   //UPDATE GUI
   //REFRESH GUI
   // Queue thread: do not attempt Join if there is no active thread

   if(cmdThread != null)
   if(cmdThread.IsAlive == true){ 
   cmdThread.Join();
   }

   //Update display if needed (indicate wait is complete)
   //UPDATE GUI
   //REFRESH GU
   }

   catch{}
}
