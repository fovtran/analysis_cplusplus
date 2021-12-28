using System;
using System.Threading;


namespace T2
{
    class Introducer
    {
    public string Message;
    public string Reply;

    public void Run(  )
    {
        Console.WriteLine (Message);
        Reply = "Hi right back!";
    }
    }
    
    class test
    {
        public static void T1()
        {
            Introducer intro = new Introducer(  );
            intro.Message = "Hello";

            new Thread (intro.Run).Start(  );

            Console.ReadLine(  );
            Console.WriteLine (intro.Reply);
        }   
    }
}