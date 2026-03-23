#!/bin/bash
set -e

# Usage:
#   bash download_tartanevent.sh /path/to/save office.zip
#   UNZIP_FILES=true bash download_tartanevent.sh /path/to/save office.zip
#   UNZIP_FILES=true DELETE_FILES=true bash download_tartanevent.sh /path/to/save office.zip

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <destination_directory> <zip_name>"
    echo "Example: $0 ./data office.zip"
    exit 1
fi

DEST_DIR="$1"
FILE="$2"
ROOT_URL="https://download.ifi.uzh.ch/rpg/web/data/iros24_rampvo/datasets/TartanEvent"

mkdir -p "$DEST_DIR"

DOWNLOAD_URL="${ROOT_URL}/${FILE}"
TARGET_FILE="$DEST_DIR/$FILE"

echo "Downloading $DOWNLOAD_URL ..."
curl -L --fail -C - -o "$TARGET_FILE" "$DOWNLOAD_URL"

echo "Verifying zip archive ..."
unzip -t "$TARGET_FILE" > /dev/null

if [ "$UNZIP_FILES" = "true" ]; then
    BASE_NAME=$(basename "$FILE" .zip)
    UNZIP_DIR="$DEST_DIR/$BASE_NAME"
    mkdir -p "$UNZIP_DIR"

    echo "Unzipping to $UNZIP_DIR ..."
    unzip -o "$TARGET_FILE" -d "$UNZIP_DIR"

    if [ "$DELETE_FILES" = "true" ]; then
        echo "Deleting $TARGET_FILE ..."
        rm -f "$TARGET_FILE"
    fi
fi

echo "Done."