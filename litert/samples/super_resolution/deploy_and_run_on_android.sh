#!/bin/bash
set -e

show_usage() {
  echo "Usage: $(basename "$0") --accelerator=<cpu|gpu|npu> [--use_gl_buffers] BAZEL_BIN_PATH"
  echo "  --accelerator:     Specifies the hardware accelerator (cpu, gpu, or npu)."
  echo "  --use_gl_buffers:  (Optional) For GPU, enables zero-copy via GL buffers."
  echo "  BAZEL_BIN_PATH:    Path to the bazel-bin directory."
}

# --- Default values ---
ACCELERATOR=""
USE_GL_BUFFERS="false"
BAZEL_BIN_PATH=""

# --- Parse command-line arguments ---
for i in "$@"; do
  case $i in
    --accelerator=*)
      ACCELERATOR="${i#*=}"
      shift
      ;;
    --use_gl_buffers)
      USE_GL_BUFFERS="true"
      shift
      ;;
    *)
      BAZEL_BIN_PATH="$i"
      shift
      ;;
  esac
done

if [[ -z "$ACCELERATOR" ]] || [[ ! "$ACCELERATOR" =~ ^(cpu|gpu|npu)$ ]] || [[ -z "$BAZEL_BIN_PATH" ]]; then
  show_usage
  exit 1
fi

if [[ "$ACCELERATOR" != "gpu" && "$USE_GL_BUFFERS" == "true" ]]; then
  echo "Warning: --use_gl_buffers is only applicable for the GPU accelerator. Ignoring."
  USE_GL_BUFFERS="false"
fi

# --- Configuration ---
LITERT_SAMPLES_PATH="litert/samples/super_resolution"
# UPDATED: Use the new loader binary and library names
LOADER_NAME="super_res_loader_${ACCELERATOR}"
LIBRARY_NAME="super_res_lib_${ACCELERATOR}.so" # Bazel prefixes .so targets with 'lib'
MODEL_NAME="super_res-float.tflite"
IMAGE_NAME="low_res_image.png"
OUTPUT_IMAGE_NAME="output_super_res.png"
DEVICE_DIR="/data/local/tmp/super_res_example"
DEVICE_LIB_DIR="/data/local/tmp/lib" # A separate directory for libraries

echo "Using accelerator: ${ACCELERATOR}"
echo "Bazel bin path:    ${BAZEL_BIN_PATH}"

# --- Prepare and Push Files ---
echo "--- Pushing files to device ---"
adb shell "mkdir -p ${DEVICE_DIR}/models ${DEVICE_DIR}/shaders ${DEVICE_DIR}/test_images ${DEVICE_LIB_DIR}"
adb push "${BAZEL_BIN_PATH}/${LITERT_SAMPLES_PATH}/${LOADER_NAME}" "${DEVICE_DIR}"
# UPDATED: Push the correct .so file
adb push "${BAZEL_BIN_PATH}/${LITERT_SAMPLES_PATH}/${LIBRARY_NAME}" "${DEVICE_LIB_DIR}"
adb push "${LITERT_SAMPLES_PATH}/models/${MODEL_NAME}" "${DEVICE_DIR}/models/"
adb push "${LITERT_SAMPLES_PATH}/shaders/." "${DEVICE_DIR}/shaders/"
adb push "${LITERT_SAMPLES_PATH}/test_images/${IMAGE_NAME}" "${DEVICE_DIR}/test_images/"

# Push shared libraries needed for execution
adb push "${BAZEL_BIN_PATH}/litert/c/libLiteRtRuntimeCApi.so" "${DEVICE_LIB_DIR}"
if [[ "$ACCELERATOR" == "gpu" ]]; then
  adb push "${BAZEL_BIN_PATH}/external/litert_gpu/jni/arm64-v8a/libLiteRtGpuAccelerator.so" "${DEVICE_LIB_DIR}"
fi
# ADDED: Push NPU libraries
if [[ "$ACCELERATOR" == "npu" ]]; then
  adb push "${BAZEL_BIN_PATH}/litert/vendors/qualcomm/dispatch/libLiteRtQualcommDispatch.so" "${DEVICE_LIB_DIR}"
  adb push -p bazel-bin/external/qairt/lib/aarch64-android/*.so "${DEVICE_LIB_DIR}"
  adb push -p bazel-bin/external/qairt/lib/hexagon-v75/unsigned/*.so "${DEVICE_LIB_DIR}"
  adb push -p bazel-bin/external/qairt/lib/hexagon-v79/unsigned/*.so "${DEVICE_LIB_DIR}"
fi

# --- Execute on Device ---
echo "--- Running the example on device ---"
# UPDATED: The command now calls the loader and passes the path to our .so library
COMMAND_TO_RUN="cd ${DEVICE_DIR} && LD_LIBRARY_PATH=${DEVICE_LIB_DIR} ./${LOADER_NAME}"
COMMAND_TO_RUN="${COMMAND_TO_RUN} ${DEVICE_LIB_DIR}/${LIBRARY_NAME}"
COMMAND_TO_RUN="${COMMAND_TO_RUN} models/${MODEL_NAME}"
COMMAND_TO_RUN="${COMMAND_TO_RUN} test_images/${IMAGE_NAME}"
COMMAND_TO_RUN="${COMMAND_TO_RUN} ${OUTPUT_IMAGE_NAME}"

if [[ "$ACCELERATOR" == "gpu" ]]; then
  COMMAND_TO_RUN="${COMMAND_TO_RUN} ${USE_GL_BUFFERS}"
fi

adb shell "${COMMAND_TO_RUN}"

# --- Pull Results ---
echo "--- Pulling results from device ---"
adb pull "${DEVICE_DIR}/${OUTPUT_IMAGE_NAME}" .
echo "Output image saved to: $(pwd)/${OUTPUT_IMAGE_NAME}"

# --- Cleanup ---
echo "--- Cleaning up files on device ---"
adb shell "rm -rf ${DEVICE_DIR} ${DEVICE_LIB_DIR}"

echo "--- Done ---"