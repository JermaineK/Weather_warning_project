# Registers a Windows scheduled task that runs the storm outlook daily and at every log-on,
# catching up any days missed while the computer was off. Run once in PowerShell:
#   powershell -ExecutionPolicy Bypass -File install_windows_task.ps1 -Out "C:\storm_outlook_data"
# Remove later with:  Unregister-ScheduledTask -TaskName StormOutlook -Confirm:$false
param(
  [string]$Out = "$env:USERPROFILE\storm_outlook_data",
  [string]$Python = (Get-Command python).Source
)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $here "run_local.py"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
# 06:45 UTC expressed in this computer's local time
$utc = [DateTime]::SpecifyKind([DateTime]::UtcNow.Date.AddHours(6).AddMinutes(45), [DateTimeKind]::Utc)
$local = $utc.ToLocalTime()
$action  = New-ScheduledTaskAction -Execute $Python -Argument "`"$script`" --out `"$Out`"" -WorkingDirectory $here
$daily   = New-ScheduledTaskTrigger -Daily -At $local
$logon   = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
            -ExecutionTimeLimit (New-TimeSpan -Hours 3) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "StormOutlook" -Action $action -Trigger $daily,$logon -Settings $settings `
  -Description "Daily storm formation outlook (resumes after shutdown)" -Force | Out-Null
Write-Host "Registered task 'StormOutlook': daily at $($local.ToString('HH:mm')) local time and at log-on. Output: $Out"
Write-Host "View the map:  python -m http.server -d `"$Out`"   then open http://localhost:8000"
