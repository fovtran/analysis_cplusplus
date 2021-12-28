[StructLayout(LayoutKind.Sequential)]
public class MySystemTime {
    public ushort wYear;
    public ushort wMonth;
    public ushort wDayOfWeek;
    public ushort wDay;
    public ushort wHour;
    public ushort wMinute;
    public ushort wSecond;
    public ushort wMilliseconds;
}
internal static class NativeMethods
{
    [DllImport("Kernel32.dll")]
    internal static extern void GetSystemTime(MySystemTime st);

    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    internal static extern int MessageBox(
        IntPtr hWnd, string lpText, string lpCaption, uint uType);
}

public class TestPlatformInvoke
{
    public static void Main()
    {
        MySystemTime sysTime = new MySystemTime();
        NativeMethods.GetSystemTime(sysTime);

        string dt;
        dt = "System time is: \n" +
              "Year: " + sysTime.wYear + "\n" +
              "Month: " + sysTime.wMonth + "\n" +
              "DayOfWeek: " + sysTime.wDayOfWeek + "\n" +
              "Day: " + sysTime.wDay;
        NativeMethods.MessageBox(IntPtr.Zero, dt, "Platform Invoke Sample", 0);
    }
}
