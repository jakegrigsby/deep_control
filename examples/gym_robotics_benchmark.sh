#!/bin/bash

echo "Benchmarking $1 on environments from the gym robotics suite"

declare -a EnvList=("FetchPickAndPlace-v1" \
        "FetchPush-v1" \
        "FetchReach-v1" \
        "FetchSlide-v1" \
        "HandManipulateBlock-v0" \
        "HandManipulateEgg-v0" \
        "HandManipulatePen-v0" \
        "HandReach-v0")


for env in "${EnvList[@]}"; do
    echo Training on env: $env
    python -m deep_control.$1 --env $env --name $1_$env
done
