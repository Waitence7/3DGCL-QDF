$root = 'C:\DGCL\3DGCL\examples\sslgraph'
Write-Host "==== $root ===="
Get-ChildItem $root -Directory -Force | ForEach-Object {
  $sum = 0
  Get-ChildItem $_.FullName -Recurse -File -Force -ErrorAction SilentlyContinue | ForEach-Object { $sum += $_.Length }
  [pscustomobject]@{ MB = [math]::Round($sum / 1MB, 1); Path = $_.FullName }
} | Sort-Object MB -Descending | ForEach-Object { '{0,10:N1} MB  {1}' -f $_.MB, $_.Path }
Write-Host "`n==== dataset subdir breakdown ===="
$ds = "$root\dataset"
if (Test-Path $ds) {
  Get-ChildItem $ds -Directory -Force | ForEach-Object {
    $sum = 0
    Get-ChildItem $_.FullName -Recurse -File -Force -ErrorAction SilentlyContinue | ForEach-Object { $sum += $_.Length }
    [pscustomobject]@{ MB = [math]::Round($sum / 1MB, 1); Path = $_.FullName }
  } | Sort-Object MB -Descending | ForEach-Object { '{0,10:N1} MB  {1}' -f $_.MB, $_.Path }
}
