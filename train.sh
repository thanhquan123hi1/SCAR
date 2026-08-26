#!/usr/bin/env bash
# train.sh - Linux/Bash 1-Click Runner
set -e

CONFIG=${1:-"training/config/models/unet_3d.yaml"}
RUN_ID=${2:-"unet3d_lge_sax_exp01"}

python run_all.py --config "$CONFIG" --run-id "$RUN_ID"
