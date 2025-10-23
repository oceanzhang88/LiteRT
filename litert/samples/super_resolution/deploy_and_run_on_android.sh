#!/bin/bash
#
# Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Script to deploy and run the super resolution sample on an Android device via ADB

# --- Default values ---
ACCELERATOR="gpu" # Default accelerator if not specified
PHONE="s25" # Default phone model
BINARY_BUILD_PATH=""

# --- Usage ---
usage() {
    echo "Usage: $0 --accelerator [gpu|npu|cpu] --phone [s24|s25] <binary_build_path>"
    echo "  --accelerator : Specify the accelerator to use (gpu, npu, or cpu). Defaults to gpu."
    echo "  --phone       : Specify the phone model (e.g., s24, s25) to select the correct NPU libraries. Defaults to s25."
    echo "  <binary_build_path> : The path to the binary build directory (e.g., bazel-bin/)."
    exit 1
}

# --- Argument Parsing ---
# Check if any arguments are provided at all.
if [ "$#" -eq 0 ]; then
    echo "Error: No arguments provided."
    usage
fi

# Initialize variables for options
USE_GL_BUFFERS=false
HOST_NPU_LIB=""
HOST_NPU_DISPATCH_LIB=""
POSITIONAL_ARGS=() # Array to hold non-option arguments

# Manually parse all arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --accelerator=*)
            ACCELERATOR="${1#*=}"
            # Validate accelerator value
            if [[ "$ACCELERATOR" != "gpu" && "$ACCELERATOR" != "npu" && "$ACCELERATOR" != "cpu" ]]; then
                echo "Error: Invalid value for --accelerator. Must be 'gpu', 'npu', or 'cpu'." >&2
                usage
            fi
            shift # past argument=value
            ;;
        --phone=*)
            PHONE="${1#*=}"
            shift # past argument=value
            ;;
        --use_gl_buffers)
            USE_GL_BUFFERS=true
            shift # past argument
            ;;
        --host_npu_lib=*)
            HOST_NPU_LIB="${1#*=}"
            shift # past argument=value
            ;;
        --host_npu_dispatch_lib=*)
            HOST_NPU_DISPATCH_LIB="${1#*=}"
            shift # past argument=value
            ;;
        -*)
            # Handle unknown options or unsupported format (like --accelerator gpu)
            echo "Error: Unknown or unsupported option format: $1" >&2
            echo "Please use the format --option=value"
            usage
            ;;
        *)
            # Assume this is a positional argument
            POSITIONAL_ARGS+=("$1")
            shift # past argument
            ;;
    esac
done

echo "Selected Accelerator: $ACCELERATOR"
echo "Use GL Buffers: $USE_GL_BUFFERS"

# Check for the correct number of positional arguments
if [ "${#POSITIONAL_ARGS[@]}" -ne 1 ]; then
    echo "Error: Incorrect number of arguments. Expected exactly one <binary_build_path>."
    usage
fi

BINARY_BUILD_PATH="${POSITIONAL_ARGS[0]}"

# Check if the binary_build_path is a valid directory.
if [ ! -d "$BINARY_BUILD_PATH" ]; then
    echo "Error: The provided binary_build_path ($BINARY_BUILD_PATH) is not a valid directory."
    exit 1
fi

# --- Configuration ---
ROOT_DIR="litert/"

PACKAGE_LOCATION="${ROOT_DIR}samples/super_resolution"
PACKAGE_NAME="super_res_standalone_${ACCELERATOR}"
OUTPUT_PATH="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}"
RUNFILES_DIR="${OUTPUT_PATH}.runfiles"

# Device paths
DEVICE_BASE_DIR="/data/local/tmp/super_res_acc_android"
DEVICE_EXEC_NAME="super_res_executable"
DEVICE_SHADER_DIR="${DEVICE_BASE_DIR}/shaders"
DEVICE_TEST_IMAGE_DIR="${DEVICE_BASE_DIR}/test_images"
DEVICE_MODEL_DIR="${DEVICE_BASE_DIR}/models"
DEVICE_NPU_LIBRARY_DIR="${DEVICE_BASE_DIR}/npu"

# Host paths
HOST_SHADER_DIR="${PACKAGE_LOCATION}/shaders"
HOST_TEST_IMAGE_DIR="${PACKAGE_LOCATION}/test_images"
HOST_MODEL_DIR="${PACKAGE_LOCATION}/models"
HOST_C_LIBRARY_DIR="${BINARY_BUILD_PATH}/litert/c/"
HOST_API_LIB_PATH="${BINARY_BUILD_PATH}/${PACKAGE_LOCATION}/super_res_lib_${ACCELERATOR}.so"
HOST_GPU_LIBRARY_DIR="${RUNFILES_DIR}/litert_gpu/jni/arm64-v8a/"

# Set NPU library path based on the --npu_dispatch_lib_path flag
if [[ -z "$HOST_NPU_LIB" ]]; then
    echo "Defaulting to QNN libraries path."
    HOST_NPU_LIB="${RUNFILES_DIR}/qairt/lib/"
fi
if [[ -z "$HOST_NPU_DISPATCH_LIB" ]]; then
    echo "Defaulting to internal dispatch library path."
    HOST_NPU_DISPATCH_LIB="${RUNFILES_DIR}/litert/litert/vendors/qualcomm/dispatch"
fi
# Qualcomm NPU library path
LD_LIBRARY_PATH="${DEVICE_NPU_LIBRARY_DIR}/"
ADSP_LIBRARY_PATH="${DEVICE_NPU_LIBRARY_DIR}/"

# --- NPU Configuration ---
QNN_STUB_LIB=""
QNN_SKEL_LIB=""
QNN_SKEL_PATH_ARCH=""
case "$PHONE" in
    's24')
        QNN_STUB_LIB="libQnnHtpV75Stub.so"
        QNN_SKEL_LIB="libQnnHtpV75Skel.so"
        QNN_SKEL_PATH_ARCH="hexagon-v75"
        ;;
    's25')
        QNN_STUB_LIB="libQnnHtpV79Stub.so"
        QNN_SKEL_LIB="libQnnHtpV79Skel.so"
        QNN_SKEL_PATH_ARCH="hexagon-v79"
        ;;
    'vst')
        QNN_STUB_LIB="libQnnHtpV69Stub.so"
        QNN_SKEL_LIB="libQnnHtpV69Skel.so"
        QNN_SKEL_PATH_ARCH="hexagon-v69"
        ;;
    *)
        echo "Error: Unsupported phone model '$PHONE'. Supported models are 's24', 's25', 'vst'." >&2
        exit 1
        ;;
esac


# --- Model Selection ---
MODEL_FILENAME="super_res-float.tflite"
TEST_IMAGE_FILENAME="low_res_image.png"

# --- Script Logic ---
echo "Starting deployment to Android device..."

# Determine executable path
HOST_EXEC_PATH="${OUTPUT_PATH}"
echo "Using output path: ${HOST_EXEC_PATH}"

if [ ! -f "${HOST_EXEC_PATH}" ]; then
    echo "Error: Executable not found at ${HOST_EXEC_PATH}"
    echo "Please ensure the project has been built and the correct path is provided."
    exit 1
fi

echo "Target device directory: ${DEVICE_BASE_DIR}"

# Create directories on device
adb shell "mkdir -p ${DEVICE_BASE_DIR}"
adb shell "mkdir -p ${DEVICE_SHADER_DIR}"
adb shell "mkdir -p ${DEVICE_TEST_IMAGE_DIR}"
adb shell "mkdir -p ${DEVICE_MODEL_DIR}"
adb shell "mkdir -p ${DEVICE_NPU_LIBRARY_DIR}"
echo "Created directories on device."

# Push executable
adb push "${HOST_EXEC_PATH}" "${DEVICE_BASE_DIR}/${DEVICE_EXEC_NAME}"
echo "Pushed executable."

# Push shaders
adb push "${HOST_SHADER_DIR}/passthrough_shader.vert" "${DEVICE_SHADER_DIR}/"
adb push "${HOST_SHADER_DIR}/super_res_compute.glsl" "${DEVICE_SHADER_DIR}/"
echo "Pushed shaders."

# Push test images
adb push "${HOST_TEST_IMAGE_DIR}/${TEST_IMAGE_FILENAME}" "${DEVICE_TEST_IMAGE_DIR}/"
echo "Pushed test images."

# Push model files
adb push "${HOST_MODEL_DIR}/${MODEL_FILENAME}" "${DEVICE_MODEL_DIR}/"
echo "Pushed super resolution model."

# Push C API shared library
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${DEVICE_BASE_DIR}/"
adb push "${HOST_C_LIBRARY_DIR}/libLiteRtRuntimeCApi.so" "${DEVICE_BASE_DIR}/"
echo "Pushed C API shared library."

# Push project-specific API library
adb push "${HOST_API_LIB_PATH}" "${DEVICE_BASE_DIR}/"
echo "Pushed project API library (${HOST_API_LIB_PATH})."

# Push GPU accelerator shared library
if [[ "$ACCELERATOR" == "gpu" ]]; then
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${DEVICE_BASE_DIR}/"
    adb push "${HOST_GPU_LIBRARY_DIR}/libLiteRtGpuAccelerator.so" "${DEVICE_BASE_DIR}/"
    echo "Pushed GPU accelerator shared library."
fi

# Push NPU dispatch library
if [[ "$ACCELERATOR" == "npu" ]]; then
adb push "${HOST_NPU_DISPATCH_LIB}/libLiteRtDispatch_Qualcomm.so" "${DEVICE_NPU_LIBRARY_DIR}/"
echo "Pushed NPU dispatch library."

# Push NPU libraries
adb push "${HOST_NPU_LIB}/aarch64-android/libQnnHtp.so" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push "${HOST_NPU_LIB}/aarch64-android/${QNN_STUB_LIB}" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push "${HOST_NPU_LIB}/aarch64-android/libQnnSystem.so" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push "${HOST_NPU_LIB}/aarch64-android/libQnnHtpPrepare.so" "${DEVICE_NPU_LIBRARY_DIR}/"
adb push "${HOST_NPU_LIB}/${QNN_SKEL_PATH_ARCH}/unsigned/${QNN_SKEL_LIB}" "${DEVICE_NPU_LIBRARY_DIR}/"
echo "Pushed NPU libraries."
fi

# Set execute permissions
adb shell "chmod +x ${DEVICE_BASE_DIR}/${DEVICE_EXEC_NAME}"
echo "Set execute permissions on device."

# --- Construct Run Command ---
MODEL_PATH_REMOTE="./models/${MODEL_FILENAME}"
IMAGE_PATH_REMOTE="./test_images/${TEST_IMAGE_FILENAME}"
OUTPUT_PATH_REMOTE="./output_image.png"

echo "Cleaning up previous run results"
adb shell "rm -f ${DEVICE_BASE_DIR}/${OUTPUT_PATH_REMOTE}"

# Base command for all backends
RUN_COMMAND_ARGS="${MODEL_PATH_REMOTE} "

# Add backend-specific args
if [[ "$ACCELERATOR" == "gpu" ]]; then
    RUN_COMMAND_ARGS+="./shaders/passthrough_shader.vert ./shaders/super_res_compute.glsl "
fi

# Add image/output paths
RUN_COMMAND_ARGS+="${IMAGE_PATH_REMOTE} ${OUTPUT_PATH_REMOTE} "

# Add GL buffer flag
if [[ "$ACCELERATOR" == "gpu" ]] && $USE_GL_BUFFERS; then
    RUN_COMMAND_ARGS+="true"
fi

RUN_COMMAND="./${DEVICE_EXEC_NAME} ${RUN_COMMAND_ARGS}"

# --- Run ---
echo ""
echo "Deployment complete."
echo "To run the super resolution sample on the device, use a command like this:"

if [[ "$ACCELERATOR" == "npu" ]]; then
    FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ADSP_LIBRARY_PATH=\"${ADSP_LIBRARY_PATH}\" ${RUN_COMMAND}"
else
    FULL_COMMAND="cd ${DEVICE_BASE_DIR} && LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\" ${RUN_COMMAND}"
fi

echo "  adb shell \"${FULL_COMMAND}\""
adb shell "${FULL_COMMAND}"

echo ""
echo "To pull the result:"
LOCAL_OUTPUT_FILE="./output_image_${ACCELERATOR}.png"
echo "  adb pull ${DEVICE_BASE_DIR}/${OUTPUT_PATH_REMOTE} ${LOCAL_OUTPUT_FILE}"
adb pull "${DEVICE_BASE_DIR}/${OUTPUT_PATH_REMOTE}" "${LOCAL_OUTPUT_FILE}"

echo "Done."