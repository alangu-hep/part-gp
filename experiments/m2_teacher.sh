#!/bin/bash

# Globals

set -x
DATADIR=${DATADIR}/JetClass
WORKDIR=${WORKDIR}
CMD=${WORKDIR}/src/main.py
DATE=$(date +%Y%m%d_%H%M%S)
MODEL_OUTPUTS=${WORKDIR}/outputs/models
METRIC_OUTPUTS=${WORKDIR}/outputs/metrics
SR_OUTPUTS=${WORKDIR}/outputs/sr_runs
LOGS=${WORKDIR}/outputs/logs

SIGNALS=(
    TTBar
    WToQQ
    HToGG
)


# Run-Specific

SIGNAL=TTBar
COMP=DL

MODEL=PELICAN
NETWORK=${WORKDIR}/wrappers/pelican_wrapper.py

DATA_FRACTION=0.1

suffix=${COMMENT:-default}

for SIGNAL in "${SIGNALS[@]}"; do

    echo "Signal: ${SIGNAL}"
    CONFIG=${WORKDIR}/data_config/JetClass/JetClass_${SIGNAL}.yaml

    set +e
    $CMD \
        --comp ${COMP} \
        --data-train \
        "${SIGNAL}:${DATADIR}/Pythia/train_100M/${SIGNAL}_*.root" \
        "ZJetsToNuNu:${DATADIR}/Pythia/train_100M/ZJetsToNuNu_*.root" \
        --data-val \
        "${SIGNAL}:${DATADIR}/Pythia/val_5M/${SIGNAL}_*.root" \
        "ZJetsToNuNu:${DATADIR}/Pythia/val_5M/ZJetsToNuNu_*.root" \
        --data-test \
        "${SIGNAL}:${DATADIR}/Pythia/test_20M/${SIGNAL}_*.root" \
        "ZJetsToNuNu:${DATADIR}/Pythia/test_20M/ZJetsToNuNu_*.root" \
        --data-config ${CONFIG} \
        --file-fraction 1 \
        --data-fraction ${DATA_FRACTION} \
        --batch-size 64 \
        --num-epochs 5 \
        --optimizer ranger \
        --start-lr 1e-3 \
        --final-lr 1e-6 \
        --lr-scheduler cos \
        --model-network ${NETWORK} \
        --model-prefix ${MODEL_OUTPUTS}/${SIGNAL}/${MODEL}/${MODEL}_${suffix}_${COMP} \
        --log ${LOGS}/${SIGNAL}/${MODEL}_${suffix}_${COMP}_${DATE}.log \
        --metrics-prefix ${METRIC_OUTPUTS}/${SIGNAL}/${MODEL}_${suffix}_${DATA_FRACTION}.root

    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Failed: ${SIGNAL}"
    else
        echo "Completed: ${SIGNAL}"
    fi

done

date +%Y%m%d_%H%M%S