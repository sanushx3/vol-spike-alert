git add .
$msg = Read-Host "Commit message (Enter for default)"
if ([string]::IsNullOrWhiteSpace($msg)) { $msg = "Progress update $(Get-Date -Format 'yyyy-MM-dd HH:mm')" }
git commit -m $msg
git push