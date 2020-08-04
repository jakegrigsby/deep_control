#!/bin/bash

echo "Benchmarking $1 on environments from the pybullet_gym suite"

declare -a EnvList=("InvertedPendulumPyBulletEnv-v0" \
    "InvertedDoublePendulumPyBulletEnv-v0" \
    "InvertedPendulumSwingupPyBulletEnv-v0" \
    "ReacherPyBulletEnv-v0" \
    "Walker2DPyBulletEnv-v0" \
    "HalfCheetahPyBulletEnv-v0" \
    "AntPyBulletEnv-v0" \
    "HopperPyBulletEnv-v0" \
    "HumanoidPyBulletEnv-v0" \
    "HumanoidFlagrunPyBulletEnv-v0" \
    "HumanoidFlagrunHarderPyBulletEnv-v0" \
    "AtlasPyBulletEnv-v0" \
    "PusherPyBulletEnv-v0" \
    "ThrowerPyBulletEnv-v0" \
    "StrikerPyBulletEnv-v0")

for env in "${EnvList[@]}"; do
    python -m deep_control.$1 --env $env --name $1_$env
done
