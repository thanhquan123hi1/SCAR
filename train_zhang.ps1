# Run Zhang (2021) Cascaded 2D-3D Training
$env:PYTHONPATH = "."
python training/train_zhang.py `
    --config training/config/models/zhang_cascaded.yaml `
    --run-id zhang_cascaded_exp01
