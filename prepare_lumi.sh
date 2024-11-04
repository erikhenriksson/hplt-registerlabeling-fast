#!/bin/bash

# Download, validate and split HPLT data package.

set -euo pipefail

source common_lumi.sh

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 PACKAGE" >&2
    echo >&2
    echo "example: $0 cleaned/eng_Latn/1.jsonl.zst" >&2
    exit 1
fi

PACKAGE="$1"
get_lock "$PACKAGE"

LOG_PATH="$LOG_BASE_DIR/${PACKAGE%%.*}.txt"
mkdir -p $(dirname "$LOG_PATH")

echo "$(date): START RUNNING prepare.sh" >> "$LOG_PATH"

cat <<EOF

------------------------------------------------------------------------------
Download source package
------------------------------------------------------------------------------
EOF

URL="$BASE_URL/$PACKAGE"
DOWNLOAD_PATH="$DOWNLOAD_BASE_DIR/$PACKAGE"
mkdir -p "$(dirname "$DOWNLOAD_PATH")"

if [ -e "$DOWNLOAD_PATH" ]; then
    echo "$DOWNLOAD_PATH exists, skipping download."
else
    echo "Downloading $URL to $DOWNLOAD_PATH ..."
    echo "$(date): START download" >> "$LOG_PATH"
    wget "$URL" -O "$DOWNLOAD_PATH" --no-clobber
    echo "$(date): END download" >> "$LOG_PATH"
    echo "Done downloading."
fi

cat <<EOF

------------------------------------------------------------------------------
Validate source package checksum
------------------------------------------------------------------------------
EOF

echo "Validating checksum for $DOWNLOAD_PATH ..."
CHECKSUM_URL="$URL.md5"
SOURCE_CHECKSUM="$(curl -s "$CHECKSUM_URL" | awk '{ print $1 }')"

echo "$(date): START checksum" >> "$LOG_PATH"
LOCAL_CHECKSUM="$(time md5sum "$DOWNLOAD_PATH" | awk '{ print $1 }')"
echo "$(date): END checksum" >> "$LOG_PATH"

if [ "$SOURCE_CHECKSUM" != "$LOCAL_CHECKSUM" ]; then
    echo "Error: checksum mismatch ($SOURCE_CHECKSUM vs $LOCAL_CHECKSUM)" >&2
    echo >&2
    echo "You may want to delete $DOWNLOAD_PATH and rerun." >&2
    exit 1
fi
echo "Done validating checksum."

cat <<EOF

------------------------------------------------------------------------------
Get source package statistics
------------------------------------------------------------------------------
EOF

STATS_PATH="$STATS_BASE_DIR/${PACKAGE%%.*}.txt"
mkdir -p "$(dirname "$STATS_PATH")"

if [ -s "$STATS_PATH" ]; then
   echo "$STATS_PATH exists, skipping taking statistics."
else
    echo "Taking statistics for $DOWNLOAD_PATH to $STATS_PATH ..."
    echo "$(date): START stats" >> "$LOG_PATH"
    time zstdcat "$DOWNLOAD_PATH" | wc > "$STATS_PATH"
    echo "$(date): END stats" >> "$LOG_PATH"
    echo "Done taking statistics."
fi

LINE_COUNT=$(awk '{ print $1 }' "$STATS_PATH")
WORD_COUNT=$(awk '{ print $2 }' "$STATS_PATH")
CHAR_COUNT=$(awk '{ print $3 }' "$STATS_PATH")
echo "$DOWNLOAD_PATH: $LINE_COUNT lines, $WORD_COUNT words, $CHAR_COUNT chars"

cat <<EOF

------------------------------------------------------------------------------
Split source package lines
------------------------------------------------------------------------------
EOF

SPLIT_DIR="$SPLIT_BASE_DIR/${PACKAGE%%.*}"
mkdir -p "$SPLIT_DIR"
SPLIT_LINES=$(((LINE_COUNT+SPLIT_PARTS-1)/SPLIT_PARTS))

if [ $(ls "$SPLIT_DIR" | wc -l) -ne 0 ]; then
    echo "$SPLIT_DIR is not empty, skipping split."
else
    echo "Splitting $DOWNLOAD_PATH into $SPLIT_DIR ..."
    echo "$(date): START split" >> "$LOG_PATH"
    time zstdcat "$DOWNLOAD_PATH" | \
	split - -l $SPLIT_LINES "$SPLIT_DIR/" -d --additional-suffix .jsonl
    echo "$(date): END split" >> "$LOG_PATH"
    echo "Done splitting."
fi

cat <<EOF

------------------------------------------------------------------------------
Validate split files
------------------------------------------------------------------------------
EOF

SPLIT_COUNT=$(find "$SPLIT_DIR" -name '*.jsonl' | wc -l)

if [ $SPLIT_COUNT -ne $SPLIT_PARTS ]; then
    echo "Error: split file count mismatch ($SPLIT_COUNT vs $SPLIT_PARTS)" >&2
    echo >&2
    echo "You may want to delete $SPLIT_DIR and rerun." >&2
    exit 1
fi

echo "Checking total lines for files in $SPLIT_DIR ..."
echo "$(date): START line count" >> "$LOG_PATH"
SPLIT_TOTAL_LINES=$(time find "$SPLIT_DIR" -name '*.jsonl' | xargs cat | wc -l)
echo "$(date): END line count" >> "$LOG_PATH"

if [ $SPLIT_TOTAL_LINES -ne $LINE_COUNT ]; then
    echo "Error: split line count mismatch ($SPLIT_TOTAL_LINES vs $LINE_COUNT)" >&2
    echo >&2
    echo "You may want to delete $SPLIT_DIR and rerun." >&2
    exit 1
fi
echo "Done checking total lines."

cat <<EOF

------------------------------------------------------------------------------
Preparation DONE, files in $SPLIT_DIR:
------------------------------------------------------------------------------
EOF
find "$SPLIT_DIR" -name '*.jsonl'

echo "$(date): END RUNNING prepare.sh" >> "$LOG_PATH"