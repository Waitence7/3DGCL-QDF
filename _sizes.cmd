@echo off
setlocal enabledelayedexpansion
for %%P in (
  "C:\DGCL\3DGCL\examples"
  "C:\DGCL\3DGCL\models"
  "C:\DGCL\3DGCL\dig"
  "C:\DGCL\3DGCL\figures"
  "C:\DGCL\3DGCL\QuantumDeepField_molecule\output"
  "C:\DGCL\3DGCL\QuantumDeepField_molecule\bench"
  "C:\DGCL\3DGCL\QuantumDeepField_molecule\dataset\QM9under7atoms_atomizationenergy_eV"
  "C:\DGCL\3DGCL\QuantumDeepField_molecule\dataset\QM9under7atoms_homolumo_eV"
  "C:\DGCL\3DGCL\QuantumDeepField_molecule\dataset\QM9full_homolumo_eV"
  "C:\DGCL\3DGCL\dataset\esol"
) do (
  echo ----- %%~P -----
  dir /a /s %%~P 2>nul | findstr /R "File(s)"
)
