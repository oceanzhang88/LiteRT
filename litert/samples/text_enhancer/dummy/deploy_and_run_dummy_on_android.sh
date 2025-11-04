#!/bin/bash

# --- Get the directory where the script is located ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# --- Parse Command-Line Arguments ---
COPY_FILES=false
for arg in "$@"; do
  if [[ "$arg" == "--copy" ]]; then
    COPY_FILES=true
    break
  fi
done

# --- Configuration ---
STANDALONE_BINARY_NAME="text_enhancer_standalone_dummy"
DUMMY_LIB_NAME="text_enhancer_lib_dummy.so"
OUTPUT_IMAGE_NAME="dummy_checkerboard_output.png"

# Local paths to the files in this (dummy) directory
LOCAL_BINARY_PATH="$SCRIPT_DIR/$STANDALONE_BINARY_NAME"
LOCAL_LIB_PATH="$SCRIPT_DIR/$DUMMY_LIB_NAME"

# Android target directory
ANDROID_DIR="/data/local/tmp/text_enhancer_dummy"

# --- Stop on any error ---
set -e

# --- Copy from bazel-out if --copy is set ---
if [[ "$COPY_FILES" == "true" ]]; then
  echo "--- --copy flag set: Updating binaries from build output ---"
  
  # Assume script is in <workspace_root>/litert/samples/text_enhancer/dummy
  WORKSPACE_ROOT="$SCRIPT_DIR/../../../.."
  
  
  BAZEL_BUILD_DIR="$WORKSPACE_ROOT/bazel-bin"
  # --- End update ---
  
  SOURCE_BINARY_PATH="$BAZEL_BUILD_DIR/litert/samples/text_enhancer/$STANDALONE_BINARY_NAME"
  SOURCE_LIB_PATH="$BAZEL_BUILD_DIR/litert/samples/text_enhancer/$DUMMY_LIB_NAME"

  # Check if source files exist before copying
  if [ ! -f "$SOURCE_BINARY_PATH" ]; then
      echo "Error: Source binary not found at $SOURCE_BINARY_PATH"
      echo "Please build first (e.g., bazel build //litert/samples/text_enhancer:text_enhancer_standalone_dummy --config=android_arm64)"
      echo "If you use a different config, please update BAZEL_OUT_CONFIG_DIR in this script."
      exit 1
  fi
  if [ ! -f "$SOURCE_LIB_PATH" ]; then
      echo "Error: Source library not found at $SOURCE_LIB_PATH"
      echo "Please build first (e.g., bazel build //litert/samples/text_enhancer:text_enhancer_lib_dummy.so --config=android_arm64)"
      echo "If you use a different config, please update BAZEL_OUT_CONFIG_DIR in this script."
      exit 1
  fi

  echo "Copying binary from: $SOURCE_BINARY_PATH"
  # -v (verbose) to show the copy, -f (force) to overwrite
  cp -vf "$SOURCE_BINARY_PATH" "$LOCAL_BINARY_PATH"

  echo "Copying library from: $SOURCE_LIB_PATH"
  cp -vf "$SOURCE_LIB_PATH" "$LOCAL_LIB_PATH"
  
  echo "--- Binary update complete ---"
else
  echo "--- Skipping binary copy (run with --copy to update from build output) ---"
fi


# --- 1. Adb Setup ---
adb wait-for-device
adb root
adb remount

echo "Creating directory on device: $ANDROID_DIR"
adb shell "mkdir -p $ANDROID_DIR"

# --- 2. Push Files ---
echo "Pushing standalone binary..."
adb push "$LOCAL_BINARY_PATH" "$ANDROID_DIR/$STANDALONE_BINARY_NAME"
adb shell "chmod +x $ANDROID_DIR/$STANDALONE_BINARY_NAME"

echo "Pushing dummy library..."
adb push "$LOCAL_LIB_PATH" "$ANDROID_DIR/$DUMMY_LIB_NAME"

echo "All files pushed."

# --- 3. Run ---
echo "Running dummy standalone..."
# The standalone expects the library path as an argument.
adb shell "cd $ANDROID_DIR && \
           ./$STANDALONE_BINARY_NAME ./$DUMMY_LIB_NAME"

echo "Run complete."

# --- 4. Pull Output ---
echo "Pulling output image: $OUTPUT_IMAGE_NAME"
# Pull the output file to the same directory as the script
adb pull "$ANDROID_DIR/$OUTPUT_IMAGE_NAME" "$SCRIPT_DIR/$OUTPUT_IMAGE_NAME"

echo "--- Dummy Run Successful! Output saved to $SCRIPT_DIR/$OUTPUT_IMAGE_NAME ---"