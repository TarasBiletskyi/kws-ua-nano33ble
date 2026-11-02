<#
.SYNOPSIS
  Log the board's USB serial output to a file for a fixed time, optionally sending commands on a schedule.
  No Python or pyserial needed (uses .NET System.IO.Ports.SerialPort).

.EXAMPLE
  # 70-second gain sweep for the pdm_rms sketch: 4 gains, 15 s each, with a SUM line before every change
  .\tools\serial_log.ps1 -Port COM4 -Seconds 70 -Out tools\out\pdm\sweep.log `
      -Schedule "1=G 20|16=S|16.5=G 40|31=S|31.5=G 60|46=S|46.5=G 80|61=S"

  # 10-minute stability run, no echo to the console
  .\tools\serial_log.ps1 -Port COM4 -Seconds 600 -Out tools\out\pdm\stability.log -Quiet

.NOTES
  Every logged line is prefixed with the host time in seconds since the port was opened. Commands sent are logged
  as ">> <command>". DTR is asserted so sketches that wait for the host (while(!Serial)) continue.
#>
param(
  [string]$Port = "COM4",
  [int]$Baud = 115200,
  [double]$Seconds = 60,
  [string]$Out = "",
  [string]$Schedule = "",   # "seconds=command|seconds=command|..."
  [switch]$Quiet
)
$ErrorActionPreference = "Stop"

$sched = @()
if ($Schedule) {
  foreach ($item in $Schedule.Split('|')) {
    $kv = $item.Split('=', 2)
    $sched += [pscustomobject]@{ At = [double]$kv[0]; Cmd = $kv[1]; Sent = $false }
  }
}

$writer = $null
if ($Out) {
  # .NET resolves relative paths against the process directory, not PowerShell's $PWD
  if (-not [IO.Path]::IsPathRooted($Out)) { $Out = [IO.Path]::GetFullPath((Join-Path $PWD.Path $Out)) }
  $dir = Split-Path $Out -Parent
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
  $writer = New-Object IO.StreamWriter($Out, $false, (New-Object Text.UTF8Encoding $false))
}

$p = New-Object System.IO.Ports.SerialPort $Port, $Baud, ([System.IO.Ports.Parity]::None), 8, ([System.IO.Ports.StopBits]::One)
$p.ReadTimeout = 500
$p.DtrEnable = $true
$p.NewLine = "`n"
$p.Open()

$sw = [Diagnostics.Stopwatch]::StartNew()
$count = 0
try {
  while ($sw.Elapsed.TotalSeconds -lt $Seconds) {
    foreach ($s in $sched) {
      if (-not $s.Sent -and $sw.Elapsed.TotalSeconds -ge $s.At) {
        $p.WriteLine($s.Cmd)
        $s.Sent = $true
        $line = "{0:F3} >> {1}" -f $sw.Elapsed.TotalSeconds, $s.Cmd
        if ($writer) { $writer.WriteLine($line) }
        if (-not $Quiet) { $line }
      }
    }
    try { $l = $p.ReadLine().TrimEnd("`r") } catch [System.TimeoutException] { continue }
    $count++
    $line = "{0:F3} {1}" -f $sw.Elapsed.TotalSeconds, $l
    if ($writer) { $writer.WriteLine($line) }
    if (-not $Quiet) { $line }
  }
}
finally {
  if ($writer) { $writer.Close() }
  $p.Close()
}
"[serial_log] $count lines in $([math]::Round($sw.Elapsed.TotalSeconds, 1)) s from $Port" + $(if ($Out) { " -> $Out" } else { "" })
