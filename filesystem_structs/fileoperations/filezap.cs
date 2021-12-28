using System.IO;
 
namespace ConsoleApplication1
{
    class Program
    {
        static void Main(string[] args)
        {
            DirectoryInfo tDir = new DirectoryInfo(@"C:\");
            string Pattern = "a";
 
            TraverseDirs(tDir, Pattern);
            Console.Read();
        }
 
        private static void TraverseDirs(DirectoryInfo dir, string Pattern) 
        {
            // Subdirs
            try         // Avoid errors such as "Access Denied"
            {
                foreach (DirectoryInfo iInfo in dir.GetDirectories())
                {
                    if (iInfo.Name.StartsWith(Pattern))
                        Console.WriteLine("Found dir:  " + iInfo.FullName);
 
                    TraverseDirs(iInfo, Pattern);
                }
            }
            catch (Exception)
            {
            }
 
            // Subfiles
            try         // Avoid errors such as "Access Denied"
            {
                foreach (FileInfo iInfo in dir.GetFiles())
                {
                    if (iInfo.Name.StartsWith(Pattern))
                        Console.WriteLine("Found file: " + iInfo.FullName);
                }
            }
            catch (Exception)
            {
            }
        }
    }
}