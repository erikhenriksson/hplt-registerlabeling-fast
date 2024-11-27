#!/bin/bash

BASE_URL="https://data.hplt-project.org/two"

DOWNLOAD_BASE_DIR="downloads"
STATS_BASE_DIR="stats"
SPLIT_BASE_DIR="splits"
LOCK_DIR="locks"
LOG_BASE_DIR="logs"
PREDICT_BASE_DIR="predictions"

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