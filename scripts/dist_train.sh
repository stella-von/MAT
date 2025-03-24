#!/usr/bin/env bash

# set the OMP_NUM_THREADS environment variable
export OMP_NUM_THREADS=16

GPUS=$1
CONFIG=$2

# usage
if [ $# -lt 2 ] ;then
    echo "usage:"
    echo "./scripts/dist_train.sh [number of gpu] [path to option file]"
    exit
fi

# Function to find an available port greater than 29500
find_free_port() {
    local port=29500  # Start checking from 29500
    while : ; do
        # Check if the port is free using `ss` or `netstat`
        if ! ss -tuln | grep -q ":$port" && ! netstat -tuln | grep -q ":$port"; then
            echo $port
            return
        fi
        port=$((port + 1))
    done
}

# Automatically find a free port
MASTER_PORT=$(find_free_port)

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
torchrun --nproc_per_node=$GPUS --master_port=$MASTER_PORT basicsr/train.py -opt $CONFIG --launcher pytorch ${@:3}
