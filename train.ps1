# train.ps1 - PowerShell 1-Click Runner for Windows
param (
    [string]$Config = "training/config/models/unet_3d.yaml",
    [string]$RunId = "unet3d_lge_sax_exp01"
)

Write-Host ">>> Starting 1-Click LGE Training Pipeline on Windows PowerShell..." -ForegroundColor Green
python run_all.py --config $Config --run-id $RunId
