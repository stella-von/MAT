#!/usr/bin/env bash

# set the OMP_NUM_THREADS environment variable
export OMP_NUM_THREADS=16

GPUS=$1
CONFIG=$2

# usage
if [ $# -ne 2 ] ;then
    echo "usage:"
    echo "./scripts/dist_test.sh [number of gpu] [path to option file]"
    exit
fi

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
torchrun --nproc_per_node=$GPUS basicsr/test.py -opt $CONFIG --launcher pytorch