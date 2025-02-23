#!/bin/bash

# Define variables
model=Gor-bepis/fact-checker-bfmtg-v2
volume=$PWD/data
tgi_port_number=8083


# Attempt to run with GPU support
echo "Attempting to run with GPU support..."
docker run --gpus all --shm-size 1g -p $tgi_port_number:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:3.1.0 \
    --model-id $model

# Check if the container is running
if [ $? -ne 0 ]; then
    echo "Failed to run with GPU. Falling back to CPU..."
    # Run without GPU support
    docker run --shm-size 1g -p $tgi_port_number:80 -v $volume:/data \
        ghcr.io/huggingface/text-generation-inference:3.1.0 \
        --model-id $model
else
    echo "Container is running with GPU support."
fi
