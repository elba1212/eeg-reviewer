param([string]$Destination = (Join-Path $env:LOCALAPPDATA 'EEGReviewer\App'))
$ErrorActionPreference = 'Stop'
$source = Join-Path $PSScriptRoot 'dist\EEGReviewer'
if (-not (Test-Path -LiteralPath (Join-Path $source 'EEGReviewer.exe'))) {
    throw 'Build the desktop app first with scripts\build_desktop.py.'
}
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
Get-ChildItem -LiteralPath $source | Copy-Item -Destination $Destination -Recurse -Force
$workspace = Join-Path $env:LOCALAPPDATA 'EEGReviewer\Workspace'
New-Item -ItemType Directory -Force -Path $workspace | Out-Null
$shortcutRoot = Join-Path ([Environment]::GetFolderPath('StartMenu')) 'Programs'
New-Item -ItemType Directory -Force -Path $shortcutRoot | Out-Null
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path $shortcutRoot 'EEG Reviewer.lnk'))
$shortcut.TargetPath = Join-Path $Destination 'EEGReviewer.exe'
$shortcut.Arguments = '--workspace "' + $workspace + '"'
$shortcut.WorkingDirectory = $Destination
$shortcut.Description = 'Local EEG sleep and epileptic activity review'
$shortcut.Save()
Write-Output ('Installed EEG Reviewer. Open it from the Start menu, or run ' + $shortcut.TargetPath)
