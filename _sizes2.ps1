$roots = @('C:\DGCL\3DGCL\examples', 'C:\DGCL\3DGCL\models')
foreach ($root in $roots) {
  Write-Host "==== $root ===="
  Get-ChildItem $root -Directory -Force | ForEach-Object {
    $sum = 0
    Get-ChildItem $_.FullName -Recurse -File -Force -ErrorAction SilentlyContinue | ForEach-Object { $sum += $_.Length }
    [pscustomobject]@{ MB = [math]::Round($sum / 1MB, 1); Path = $_.FullName }
  } | Sort-Object MB -Descending | ForEach-Object { '{0,10:N1} MB  {1}' -f $_.MB, $_.Path }
  $rootFiles = 0
  Get-ChildItem $root -File -Force | ForEach-Object { $rootFiles += $_.Length }
  '{0,10:N1} MB  (files directly in {1})' -f ($rootFiles / 1MB), $root
}
