#!/bin/bash

export PYTHONPATH=.
export HF_HOME="/Volumes/DOCKER/ml-models"
export HF_TOKEN=$(cat hf-token)

export RUNNER="python "

if [ "${2}" == "debug" ]
then
    export RUNNER="python -m debugpy --listen 5678 --wait-for-client " 
    echo "Debug mode; attach via VS Code; debug start command is ${RUNNER}"
fi 

${RUNNER} ${1}