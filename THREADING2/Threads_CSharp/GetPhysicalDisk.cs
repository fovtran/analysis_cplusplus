Add-Type -TypeDefinition @”
using System;
using Microsoft.Win32.SafeHandles;
using System.IO;
using System.Runtime.InteropServices;

public class GetDisk
{
private const uint IoctlVolumeGetVolumeDiskExtents = 0x560000;

[StructLayout(LayoutKind.Sequential)]
public struct DiskExtent
{
public int DiskNumber;
public Int64 StartingOffset;
public Int64 ExtentLength;
}

[StructLayout(LayoutKind.Sequential)]
public struct DiskExtents
{
public int numberOfExtents;
public DiskExtent first;
}

[DllImport(“Kernel32.dll”, SetLastError = true, CharSet = CharSet.Auto)]
private static extern SafeFileHandle CreateFile(
string lpFileName,
[MarshalAs(UnmanagedType.U4)] FileAccess dwDesiredAccess,
[MarshalAs(UnmanagedType.U4)] FileShare dwShareMode,
IntPtr lpSecurityAttributes,
[MarshalAs(UnmanagedType.U4)] FileMode dwCreationDisposition,
[MarshalAs(UnmanagedType.U4)] FileAttributes dwFlagsAndAttributes,
IntPtr hTemplateFile);

[DllImport(“Kernel32.dll”, SetLastError = false, CharSet = CharSet.Auto)]
private static extern bool DeviceIoControl(
SafeFileHandle hDevice,
uint IoControlCode,
[MarshalAs(UnmanagedType.AsAny)] [In] object InBuffer,
uint nInBufferSize,
ref DiskExtents OutBuffer,
int nOutBufferSize,
ref uint pBytesReturned,
IntPtr Overlapped
);

public static string GetPhysicalDriveString(string path)
{
//clean path up
path = path.TrimEnd(‘\’);
if (!path.StartsWith(@”\.\”))
path = @”\.\” + path;

SafeFileHandle shwnd = CreateFile(path, FileAccess.Read, FileShare.Read | FileShare.Write, IntPtr.Zero, FileMode.Open, 0,
IntPtr.Zero);
if (shwnd.IsInvalid)
{
//Marshal.ThrowExceptionForHR(Marshal.GetLastWin32Error());
Exception e = Marshal.GetExceptionForHR(Marshal.GetLastWin32Error());
}

var bytesReturned = new uint();
var de1 = new DiskExtents();
bool result = DeviceIoControl(shwnd, IoctlVolumeGetVolumeDiskExtents, IntPtr.Zero, 0, ref de1,
Marshal.SizeOf(de1), ref bytesReturned, IntPtr.Zero);
shwnd.Close();
if(result)
return @”\.\PhysicalDrive” + de1.first.DiskNumber;
return null;
}
}

“@

function Get-Disk2Mount
{

$MountInfos = Get-WmiObject Win32_Volume -Filter "DriveType='3'" | select Label,Name,DeviceID,Capacity,SystemName

$MountPoints = @()

foreach($info in $MountInfos)
{
$volumeID = (($info.DeviceID -split '\\') -match "Volume*").trim()

$PhysicalDisk = ([getDisk]::GetPhysicalDriveString($volumeID) -split '\\')[-1]

$mountPoint = New-Object -TypeName psobject

$mountPoint | Add-Member -MemberType NoteProperty -Name 'Label' -Value $($info.label)
$mountPoint | Add-Member -MemberType NoteProperty -Name 'Name' -Value $info.Name
$mountPoint | Add-Member -MemberType NoteProperty -Name 'PhysicalDisk' -Value $PhysicalDisk
$mountPoint | Add-Member -MemberType NoteProperty -Name 'Size(GB)' -Value $([math]::round($info.Capacity/1GB))
$mountPoint | Add-Member -MemberType NoteProperty -Name 'HostName' -Value $info.SystemName

$MountPoints += @($mountPoint)

}

return $MountPoints

}