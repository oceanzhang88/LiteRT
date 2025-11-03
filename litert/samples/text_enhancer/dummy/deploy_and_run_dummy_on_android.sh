#!/bin/bash
#
# A dedicated script to deploy and run the 'dummy' standalone backend
# on a connected Android device.
#
set -e

echo "--- Deploying Super-Res Standalone (Dummy) ---"

# --- 1. Define Targets ---
TARGET_APP_DIR="/data/local/tmp/super_res_dummy"
TARGET_APP="super_res_standalone_dummy"
TARGET_LIB="super_res_lib_dummy.so"
OUTPUT_FILE="dummy_checkerboard_output.png"

# --- 3. Setup Remote Directory ---
echo "Creating remote directory: ${TARGET_APP_DIR}"
adb shell "rm -rf ${TARGET_APP_DIR}"
adb shell "mkdir -p ${TARGET_APP_DIR}"

# --- 4. Push Files to Device ---
echo "Pushing main executable and library..."
adb push "bazel-bin/litert/samples/super_resolution/${TARGET_APP}" "${TARGET_APP_DIR}"
adb push "bazel-bin/litert/samples/super_resolution/${TARGET_LIB}" "${TARGET_APP_DIR}"

# --- 5. Run the Executable ---
echo "Running ${TARGET_APP} on device..."
# The dummy executable takes no arguments
adb shell "cd ${TARGET_APP_DIR} && LD_LIBRARY_PATH=. ./${TARGET_APP}"

# --- 6. Pull the Result ---
echo "Pulling output file: ${OUTPUT_FILE}"
adb pull "${TARGET_APP_DIR}/${OUTPUT_FILE}" .

echo "--- Done! ---"
echo "Check for '${OUTPUT_FILE}' in your current directory."