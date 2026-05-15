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
  'C:\DGCL\3DGCL\dataset\esol'
)
foreach ($p in $paths) {
    if (-not (Test-Path $p)) { continue }
    $lines = robocopy $p NUL /L /S /BYTES /NFL /NDL /NJH /NC /R:0 /W:0 /XJ 2>$null
    foreach ($line in $lines) {
        $t = "$line".Trim()
        if ($t.StartsWith('Bytes :')) {
            $parts = ($t -replace 'Bytes\s*:', '').Trim() -split '\s+'
            $b = [int64]$parts[0]
            "{0,10:N1} MB  {1}" -f ($b / 1MB), $p
            break
        }
    }
}
