using System;
using System.Threading;
using System.Collections.Generic;

namespace T10
{
    class test
    {
        static List <string> list = new List <string>(  );

        static void T1(  )
        {
        new Thread (AddItems).Start(  );
        new Thread (AddItems).Start(  );
        }

        static void AddItems(  )
        {
        for (int i = 0; i < 100; i++)
            lock (list)
            //if (!myList.Contains (newItem)) myList.Add (newItem);
            list.Add ("Item " + list.Count);

        string[] items;
        lock (list) items = list.ToArray(  );
        foreach (string s in items) Console.WriteLine (s);
        }
    }
}