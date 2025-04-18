#!/bin/bash

BASE_URL="https://data.hplt-project.org/two"

DOWNLOAD_BASE_DIR="downloads"
STATS_BASE_DIR="stats"
SPLIT_BASE_DIR="/scratch/project_462000353/ehenriks/HPLT-REGISTER-SAMPLES/splits"
LOCK_DIR="locks_xlmr_large"
LOG_BASE_DIR="logs_xlmr_large"
PREDICT_BASE_DIR="/scratch/project_462000353/ehenriks/HPLT-REGISTER-SAMPLES/predictions"

SPLIT_PARTS=8

get_lock() {
    local package="$1"
    local lockfile="$LOCK_DIR/${package//\//_}"
    mkdir -p "$LOCK_DIR"
    exec {lockfd}> "$lockfile"    # Open file descriptor
    if ! flock -n $lockfd; then
        echo "Already running (lockfile $lockfile), exiting."
        exit 1
    else
        echo "$$" > "$lockfile"    # Write PID to lock file
        trap 'rm -f "'"$lockfile"'"' EXIT    # Remove lock file on exit
    fi
}