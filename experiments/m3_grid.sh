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
FIGURES=${WORKDIR}/figures

SIGNALS=(
    TTBar
    WToQQ
    HToGG
)

# Run-Specific

COMP=KD

MODEL=ParT
NETWORK=${WORKDIR}/wrappers/part_wrapper.py
TEACHER_NETWORK=${WORKDIR}/wrappers/pelican_wrapper.py
TEACHER_NAME=PELICAN

DATA_FRACTION=0.01

TEMPS=(
    1
    3
    5
    7
)

suffix=${COMMENT:-default}

for SIGNAL in "${SIGNALS[@]}"; do

    echo "Signal: ${SIGNAL}"
    CONFIG=${WORKDIR}/data_config/JetClass/JetClass_${SIGNAL}.yaml
    TEACHER_PATH=${WORKDIR}/outputs/models/${SIGNAL}/${TEACHER_NAME}/${TEACHER_NAME}_M2-TEACHER_DL_epoch-4_state.pt

    set +e
    for TEMPERATURE in "${TEMPS[@]}"; do
    
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
            --final-lr 1e-3 \
            --lr-scheduler flat+decay \
            --model-network ${NETWORK} \
            --model-prefix ${MODEL_OUTPUTS}/${SIGNAL}/${MODEL}/KD_GRID/${MODEL}_${suffix}_${COMP}_${TEACHER_NAME}_${TEMPERATURE} \
            --log ${LOGS}/${SIGNAL}/${MODEL}_${suffix}_${COMP}_${TEACHER_NAME}_${TEMPERATURE}_${DATE}.log \
            --metrics-prefix ${METRIC_OUTPUTS}/${SIGNAL}/${MODEL}_${suffix}_${DATA_FRACTION}.root \
            --teacher-network ${TEACHER_NETWORK} \
            --teacher-path ${TEACHER_PATH} \
            --teacher-prefix ${TEACHER_NAME} \
            --kd-temp ${TEMPERATURE} \
            --figures-prefix ${FIGURES}/${SIGNAL}/${MODEL}/KD_GRID/${MODEL}_${suffix}_${COMP}_${TEACHER_NAME}_${TEMPERATURE}
            
    done

    set -e
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Failed: ${SIGNAL}"
    else
        echo "Completed: ${SIGNAL}"
    fi

done

COMP=DL

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
        --final-lr 1e-3 \
        --lr-scheduler flat+decay \
        --model-network ${NETWORK} \
        --model-prefix ${MODEL_OUTPUTS}/${SIGNAL}/${MODEL}/KD_GRID/${MODEL}_${suffix}_${COMP} \
        --log ${LOGS}/${SIGNAL}/${MODEL}_${suffix}_${COMP}_${DATE}.log \
        --metrics-prefix ${METRIC_OUTPUTS}/${SIGNAL}/${MODEL}_${suffix}_${DATA_FRACTION}.root \
        --figures-prefix ${FIGURES}/${SIGNAL}/${MODEL}/KD_GRID/${MODEL}_${suffix}_${COMP}

    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Failed: ${SIGNAL}"
    else
        echo "Completed: ${SIGNAL}"
    fi

done