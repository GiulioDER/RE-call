param(
  [switch]$WhatIf,
  [string]$TaskName = "recall-mcp-sweep",
  [string]$Repo = (Join-Path $env:USERPROFILE "Documents\recall")
)

# Register (or re-register) the hourly task that closes MCP servers on VPS2 whose client is gone.
# The body is scripts/session-mcp-sweep-task.sh, read from origin/master at every run, so this
# registration is the only thing that ever needs doing by hand.
#
# ── WHY EACH SETTING IS WHAT IT IS ───────────────────────────────────────────────────────────────
#
# LogonType Interactive, NOT S4U
#   The sweep is safe only if it can read the command line of every local ssh transport, because
#   a transport's launch ID is what marks its server as held. From another logon session Windows
#   may return that command line empty, and the sweep would then read a live transport as gone.
#   (It refuses when it sees one, but a guard is a backstop, not a plan.) Running inside the
#   signed-in session sees exactly what the Claude sessions see. Nothing is lost while signed out:
#   with no session running, no server can leak, and the first tick after sign-in closes whatever
#   was left.
#
# Hourly, StartWhenAvailable
#   A leaked server costs 15 to 100 MB on a host that runs the live trading services (measured
#   2026-09-26), so an hour is the bound. A tick missed while the machine slept runs on wake,
#   which is exactly when the transports of a sleeping session have died.
#
# ExecutionTimeLimit PT10M, MultipleInstances IgnoreNew
#   A sweep is two ssh calls. A hung one must not suppress the next tick for the default 72 hours.

$ErrorActionPreference = "Stop"

$bash = "C:\Program Files\Git\usr\bin\bash.exe"
if (-not (Test-Path $bash)) { throw "Git bash not found: $bash" }
if (-not (Test-Path (Join-Path $Repo ".git"))) { throw "not a recall checkout: $Repo" }

$repoPosix = ($Repo -replace '\\', '/')
$body = "git -C '$repoPosix' fetch -q origin master; " +
        "git -C '$repoPosix' show origin/master:scripts/session-mcp-sweep-task.sh | bash -s -- '$repoPosix'"
$action = New-ScheduledTaskAction -Execute $bash -Argument ("-lc `"$body`"")

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddMinutes(17)
$trigger.Repetition = (New-ScheduledTaskTrigger -Once -At (Get-Date) `
  -RepetitionInterval (New-TimeSpan -Hours 1)).Repetition

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
  -LogonType Interactive -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet `
  -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
  -MultipleInstances IgnoreNew `
  -StartWhenAvailable `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -Hidden

if ($WhatIf) {
  "WOULD REGISTER $TaskName"
  "  action:    $bash -lc `"$body`""
  "  trigger:   hourly, at :17"
  "  principal: $env:USERDOMAIN\$env:USERNAME, Interactive, Limited"
  "  log:       $env:USERPROFILE\.claude\logs\mcp-sweep.log"
  exit 0
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
  -Principal $principal -Settings $settings -Force | Out-Null
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State | Format-List
