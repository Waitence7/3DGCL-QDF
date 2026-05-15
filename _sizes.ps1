$paths = @(
  'C:\DGCL\3DGCL\examples',
  'C:\DGCL\3DGCL\models',
  'C:\DGCL\3DGCL\dig',
  'C:\DGCL\3DGCL\figures',
  'C:\DGCL\3DGCL\QuantumDeepField_molecule\output',
  'C:\DGCL\3DGCL\QuantumDeepField_molecule\bench',
  'C:\DGCL\3DGCL\QuantumDeepField_molecule\dataset\QM9under7atoms_atomizationenergy_eV',
  'C:\DGCL\3DGCL\QuantumDeepField_molecule\dataset\QM9under7atoms_homolumo_eV',
  'C:\DGCL\3DGCL\QuantumDeepField_molecule\dataset\QM9full_homolumo_eV',
  'C:\DGCL\3DGCL\dataset\esol',
  'C:\DGCL\3DGCL\dataset'
)
foreach ($p in $paths) {
  if (-not (Test-Path $p)) { continue }
  $sum = 0
  Get-ChildItem $p -Recurse -File -Force -ErrorAction SilentlyContinue | ForEach-Object { $sum += $_.Length }
  $mb = $sum / 1MB
  '{0,12:N1} MB  {1}' -f $mb, $p
}
