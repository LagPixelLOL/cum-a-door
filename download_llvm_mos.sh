#!/bin/bash

# A script to download and extract the latest llvm-mos SDK.

# --- Configuration ---
URL="https://github.com/llvm-mos/llvm-mos-sdk/releases/latest/download/llvm-mos-linux.tar.xz"
ARCHIVE_FILE="llvm-mos-linux.tar.xz"
DEST_DIR="llvm-mos"

# --- Script Safety ---
# Exit immediately if a command exits with a non-zero status. This is the "fail short" part.
set -e
# Treat unset variables as an error.
set -u
# If any command in a pipeline fails, the entire pipeline's exit status will be that of the failed command.
set -o pipefail

# --- Cleanup Function ---
# This function is registered with "trap" and will be called if the script exits with an error.
cleanup_on_failure() {
  echo "❌ An error occurred. Cleaning up..."
  rm -f "$ARCHIVE_FILE"
  rm -rf "$DEST_DIR"
  echo "Cleanup complete. No files or directories were left behind."
}

# Register the cleanup function to run on any script error (non-zero exit code).
trap cleanup_on_failure ERR

# --- Main Execution ---
# Prepare the environment by removing any old versions.
echo "Preparing the environment..."
rm -rf "$DEST_DIR" "$ARCHIVE_FILE"

# Download the file using curl.
echo "Downloading llvm-mos SDK from \"$URL\"..."
curl -Lo "$ARCHIVE_FILE" "$URL"
echo "Download complete."

# Create the destination directory and extract the archive into it.
echo "Creating destination directory \"$DEST_DIR\"..."
mkdir -p "$DEST_DIR"

echo "Extracting \"$ARCHIVE_FILE\"..."
tar -xf "$ARCHIVE_FILE" -C "$DEST_DIR" --strip-components=1
echo "Extraction complete."

# Clean up the downloaded compressed file on success.
echo "Cleaning up the compressed file..."
rm "$ARCHIVE_FILE"

# --- Success ---
# Disable the trap so it doesn't run the cleanup function on a successful exit.
trap - ERR

echo "✅ llvm-mos SDK was successfully installed into the \"$DEST_DIR\" directory."
