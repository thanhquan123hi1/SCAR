#!/bin/bash
export PYTHONPATH="."
python training/train_zhang.py \
    --config training/config/models/zhang_cascaded.yaml \
    --run-id zhang_cascaded_exp01
