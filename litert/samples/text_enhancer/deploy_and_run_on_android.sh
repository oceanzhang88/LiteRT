#!/bin/bash

# Deploys and runs the Text Enhancer sample on a connected Android device.
#
# Prerequisites:
# 1. Android device connected with 'adb'.
# 2. 'bazel build' has been run for the desired target
#    (e.g., :text_enhancer_standalone_npu).
#
# Usage:
#   ./deploy_and_run_on_android.sh [OPTIONS] <bazel_bin_path>
#
#   <bazel_bin_path>: Path to the 'bazel-bin' directory.
#
#   OPTIONS:
#     --accelerator=cpu|gpu|npu  (Default: cpu)
#     --preprocessor=cpu|vulkan  (Default: cpu)
#     --save_preprocessed        (Default: false)
#     --phone=s25|s23            (Default: s25. Used for NPU lib path & model)
#

set -e

# --- Default Values ---
ACCELERATOR_NAME="cpu"
PREPROCESSOR_TYPE="cpu"
SAVE_PREPROCESSED=false
PHONE_MODEL="s25"
HOST_NPU_LIB=""
HOST_NPU_DISPATCH_LIB=""
POSITIONAL_ARGS=() # Array to hold non-option arguments

# --- Helper Function ---
show_help() {
    echo "Usage: ./deploy_and_run_on_android.sh [OPTIONS] <bazel_bin_path>"
    echo ""
    echo "Deploys and runs the Text Enhancer sample on a connected Android device."
    echo ""
    echo "Options:"
    echo "  --accelerator=cpu|gpu|npu  (Default: cpu)"
    echo "  --preprocessor=cpu|vulkan  (Default: cpu)"
    echo "  --save_preprocessed        (Default: false)"
    echo "  --phone=s25|vst            (Default: s25. Used for NPU lib path & model)"
    echo "  --host_npu_lib=<path>          (Optional. Overrides default QNN host lib path)"
    echo "  --host_npu_dispatch_lib=<path> (Optional. Overrides default dispatch host lib path)"
    echo "  --help                     Show this help message"
}

# --- Argument Parsing ---
if [ "$#" -eq 0 ]; then
    echo "Error: No arguments provided."
    show_help
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --accelerator=*)
        ACCELERATOR_NAME="${1#*=}"
        # Validate accelerator value
        if [[ "$ACCELERATOR_NAME" != "gpu" && "$ACCELERATOR_NAME" != "npu" && "$ACCELERATOR_NAME" != "cpu" ]]; then
            echo "Error: Invalid value for --accelerator. Must be 'gpu', 'npu', or 'cpu'." >&2
            show_help
            exit 1
        fi
        shift
        ;;
        --preprocessor=*)
        PREPROCESSOR_TYPE="${1#*=}"
        shift
        ;;
        --save_preprocessed)
        SAVE_PREPROCESSED=true
        shift
        ;;
        --phone=*)
        PHONE_MODEL="${1#*=}"
        shift
        ;;
        --host_npu_lib=*)
        HOST_NPU_LIB="${1#*=}"
        shift
        ;;
        --host_npu_dispatch_lib=*)
        HOST_NPU_DISPATCH_LIB="${1#*=}"
        shift
        ;;
        --help)
        show_help
        exit 0
        ;;
        -*)
        echo "Error: Unknown or unsupported option format: $1" >&2
        echo "Please use the format --option=value"
        show_help
        exit 1
        ;;
        *)
        # Assume this is a positional argument
        POSITIONAL_ARGS+=("$1")
        shift
        ;;
    esac
done

# Check for the correct number of positional arguments
if [ "${#POSITIONAL_ARGS[@]}" -ne 1 ]; then
    echo "Error: Incorrect number of arguments. Expected exactly one <bazel_bin_path>."
    show_help
    exit 1
fi

BAZEL_BIN_PATH="${POSITIONAL_ARGS[0]}"
BAZEL_BIN_PATH=$(echo "$BAZEL_BIN_PATH" | sed 's:/*$::')

# Check if the BAZEL_BIN_PATH is a valid directory.
if [ ! -d "$BAZEL_BIN_PATH" ]; then
    echo "Error: The provided bazel_bin_path ($BAZEL_BIN_PATH) is not a valid directory."
    exit 1
fi

echo "Selected Accelerator: $ACCELERATOR_NAME"
echo "Selected Pre-processor: $PREPROCESSOR_TYPE"
echo "Save Preprocessed Image: $SAVE_PREPROCESSED"
echo "Selected Phone Model: $PHONE_MODEL"


# --- NPU Configuration ---
QNN_STUB_LIB=""
QNN_SKEL_LIB=""
QNN_SKEL_PATH_ARCH=""
case "$PHONE_MODEL" in
    'vst') # Assuming s23 -> 8 Gen 2 -> V75
        QNN_STUB_LIB="libQnnHtpV69Stub.so"
        QNN_SKEL_LIB="libQnnHtpV69Skel.so"
        QNN_SKEL_PATH_ARCH="hexagon-v69"
        echo "Configuring for XRGEN2+ using V69 NPU libraries."
        ;;
    's25') # Assuming s25 -> 8 Gen 4 -> V79
        QNN_STUB_LIB="libQnnHtpV79Stub.so"
        QNN_SKEL_LIB="libQnnHtpV79Skel.so"
        QNN_SKEL_PATH_ARCH="hexagon-v79"
        echo "Configuring for S25 (8 Gen 4) using V79 NPU libraries."
        ;;
    *)
        if [ "$ACCELERATOR_NAME" == "npu" ]; then
            echo "Error: Unsupported phone model '$PHONE_MODEL' for NPU. Supported models are 's23', 's25'." >&2
            exit 1
        fi
        ;;
esac

# --- Model Selection ---
# Select the correct model based on accelerator and phone
MODEL_BASENAME="super_res-float_gpu.tflite" # Default for CPU/GPU
if [[ "$ACCELERATOR_NAME" == "npu" ]]; then
    if [[ "$PHONE_MODEL" == "vst" ]]; then
        MODEL_BASENAME="super_res-float_npu_vst.tflite"
    elif [[ "$PHONE_MODEL" == "s25" ]]; then
        # MODEL_BASENAME="real_x4v3_8750-float.tflite"
        MODEL_BASENAME="super_res-float_npu.tflite"
    fi
    echo "Using NPU Model: $MODEL_BASENAME"
fi


# --- Define Paths ---
TARGET_DIR="/data/local/tmp/text_enhancer_android"

# Paths relative to bazel-bin
EXECUTABLE_REL_PATH="litert/samples/text_enhancer/text_enhancer_standalone_${ACCELERATOR_NAME}"
LIB_REL_PATH="litert/samples/text_enhancer/text_enhancer_lib_${ACCELERATOR_NAME}.so"
RUNTIME_LIB_REL_PATH="litert/c/libLiteRtRuntimeCApi.so"

# Paths relative to project root (now inside text_enhancer)
SHADER_REL_PATH="litert/samples/text_enhancer/shaders/crop_resize.spv"
IMAGE_REL_PATH="litert/samples/text_enhancer/test_images/low_res_image.png"
MODEL_DIR_REL_PATH="litert/samples/text_enhancer/models"

# GPU-specific assets
ROOT_DIR="litert/"
PACKAGE_LOCATION="${ROOT_DIR}samples/text_enhancer"
PACKAGE_NAME="text_enhancer_standalone_${ACCELERATOR_NAME}"
HOST_GPU_LIBRARY_DIR="${PACKAGE_LOCATION}/libs"
# HOST_GPU_LIBRARY_DIR="${BAZEL_BIN_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/litert_gpu/jni/arm64-v8a"

# NPU-specific assets
DEVICE_NPU_LIBRARY_DIR="${TARGET_DIR}/npu"
LD_LIBRARY_PATH_ON_DEVICE="${DEVICE_NPU_LIBRARY_DIR}/"
ADSP_LIBRARY_PATH_ON_DEVICE="${DEVICE_NPU_LIBRARY_DIR}/"

# Set NPU library paths on host
if [[ -z "$HOST_NPU_LIB" ]]; then
    echo "Defaulting to QNN libraries path."
    HOST_NPU_LIB="${BAZEL_BIN_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/qairt/lib/"
fi
if [[ -z "$HOST_NPU_DISPATCH_LIB" ]]; then
    echo "Defaulting to internal dispatch library path."
    HOST_NPU_DISPATCH_LIB="${BAZEL_BIN_PATH}/${PACKAGE_LOCATION}/${PACKAGE_NAME}.runfiles/litert/litert/vendors/qualcomm/dispatch"
fi

# Basenames for on-device paths
EXECUTABLE_NAME_ON_DEVICE=$(basename "$EXECUTABLE_REL_PATH")
LIB_NAME_ON_DEVICE=$(basename "$LIB_REL_PATH")
RUNTIME_LIB_NAME_ON_DEVICE=$(basename "$RUNTIME_LIB_REL_PATH")
IMAGE_BASENAME=$(basename "$IMAGE_REL_PATH")

# --- MODIFIED: Base output name and directory for 10-run test ---
OUTPUT_IMAGE_BASENAME="output"
OUTPUT_RUN_DIR_ON_DEVICE="output_run_images"
# --- End Modification ---

OUTPUT_PREPROCESSED_IMAGE="preprocessed_output.png"


# --- Start Deployment ---
echo "Starting deployment to Android device..."
echo "Using output path: $BAZEL_BIN_PATH/$EXECUTABLE_REL_PATH"

adb shell "mkdir -p $TARGET_DIR/models"
adb shell "mkdir -p $TARGET_DIR/test_images"
adb shell "mkdir -p $TARGET_DIR/shaders"
adb shell "mkdir -p $DEVICE_NPU_LIBRARY_DIR"
echo "Created directories on device."

adb push "$BAZEL_BIN_PATH/$EXECUTABLE_REL_PATH" "$TARGET_DIR/$EXECUTABLE_NAME_ON_DEVICE"
echo "Pushed executable."

if [ "$PREPROCESSOR_TYPE" == "vulkan" ]; then
    if [ -f "$SHADER_REL_PATH" ]; then
        adb push "$SHADER_REL_PATH" "$TARGET_DIR/shaders/"
        echo "Pushed Vulkan pre-processor shader (crop_resize.spv)."
    else
        echo "Warning: Vulkan shader $SHADER_REL_PATH not found. Skipping."
    fi
fi

adb push "$IMAGE_REL_PATH" "$TARGET_DIR/test_images/"
echo "Pushed test images."

adb push "$MODEL_DIR_REL_PATH/$MODEL_BASENAME" "$TARGET_DIR/models/"
echo "Pushed text enhancer model ($MODEL_BASENAME)."

adb push "$BAZEL_BIN_PATH/$RUNTIME_LIB_REL_PATH" "$TARGET_DIR/$RUNTIME_LIB_NAME_ON_DEVICE"
echo "Pushed C API shared library."

adb push "$BAZEL_BIN_PATH/$LIB_REL_PATH" "$TARGET_DIR/$LIB_NAME_ON_DEVICE"
echo "Pushed project API library ($BAZEL_BIN_PATH/$LIB_REL_PATH)."

# --- Push Accelerator-Specific Libs ---
LD_PATH="$TARGET_DIR" # Default path for CPU/GPU

if [ "$ACCELERATOR_NAME" == "gpu" ]; then
    adb push "$HOST_GPU_LIBRARY_DIR/libLiteRtOpenClAccelerator.so" "$TARGET_DIR/"
    echo "Pushed GPU accelerator library."
fi

if [ "$ACCELERATOR_NAME" == "npu" ]; then
    echo "Pushing NPU libraries..."
    adb push "${HOST_NPU_DISPATCH_LIB}/libLiteRtDispatch_Qualcomm.so" "${DEVICE_NPU_LIBRARY_DIR}/"
    echo "Pushed NPU dispatch library."

    adb push "${HOST_NPU_LIB}/aarch64-android/libQnnHtp.so" "${DEVICE_NPU_LIBRARY_DIR}/"
    adb push "${HOST_NPU_LIB}/aarch64-android/${QNN_STUB_LIB}" "${DEVICE_NPU_LIBRARY_DIR}/"
    adb push "${HOST_NPU_LIB}/aarch64-android/libQnnSystem.so" "${DEVICE_NPU_LIBRARY_DIR}/"
    adb push "${HOST_NPU_LIB}/aarch64-android/libQnnHtpPrepare.so" "${DEVICE_NPU_LIBRARY_DIR}/"
    adb push "${HOST_NPU_LIB}/${QNN_SKEL_PATH_ARCH}/unsigned/${QNN_SKEL_LIB}" "${DEVICE_NPU_LIBRARY_DIR}/"
    echo "Pushed QNN NPU libraries."

    # Update LD_PATH to include NPU libs and target dir
    LD_PATH="$DEVICE_NPU_LIBRARY_DIR:$TARGET_DIR"
fi

adb shell "chmod +x $TARGET_DIR/$EXECUTABLE_NAME_ON_DEVICE"
echo "Set execute permissions on device."

# --- MODIFIED: Cleanup for 10-run test ---
echo "Cleaning up previous run results"
adb shell "rm -f $TARGET_DIR/$OUTPUT_IMAGE_BASENAME.png" # Old single file
adb shell "rm -f $TARGET_DIR/$OUTPUT_PREPROCESSED_IMAGE"
adb shell "rm -rf $TARGET_DIR/$OUTPUT_RUN_DIR_ON_DEVICE" # New directory
# --- End Modification ---


# --- Prepare Run Command ---
PREPROCESSOR_FLAG="--preprocessor=$PREPROCESSOR_TYPE"
SHADER_FLAG=""
if [ "$PREPROCESSOR_TYPE" == "vulkan" ]; then
    SHADER_FLAG="--shader_path=shaders/crop_resize.spv"
fi

SAVE_PREPROCESSED_FLAG=""
if [ "$SAVE_PREPROCESSED" = true ]; then
    SAVE_PREPROCESSED_FLAG="--save_preprocessed=true"
fi

# --- MODIFIED: The 5th argument is now a "base name" for the 10 outputs ---
RUN_COMMAND="./$EXECUTABLE_NAME_ON_DEVICE \
     ./$LIB_NAME_ON_DEVICE \
     ./models/$MODEL_BASENAME \
     ./test_images/$IMAGE_BASENAME \
     ./$OUTPUT_IMAGE_BASENAME \
     $PREPROCESSOR_FLAG \
     $SHADER_FLAG \
     $SAVE_PREPROCESSED_FLAG \
     --platform=android \
     "
# --- End Modification ---

if [[ "$ACCELERATOR_NAME" == "npu" ]]; then
    FULL_COMMAND="cd $TARGET_DIR && LD_LIBRARY_PATH=\"$LD_PATH\" ADSP_LIBRARY_PATH=\"$ADSP_LIBRARY_PATH_ON_DEVICE\" $RUN_COMMAND"
else
    FULL_COMMAND="cd $TARGET_DIR && LD_LIBRARY_PATH=\"$LD_PATH\" $RUN_COMMAND"
fi

echo ""
echo "Deployment complete."
echo "To run the text enhancer sample on the device, use a command like this:"
echo "  adb shell \"$FULL_COMMAND\""

# --- Execute the command ---
adb shell "$FULL_COMMAND"

# --- MODIFIED: Pull Results ---
echo ""
echo "Pulling results from 10-run benchmark..."

# Define a local directory to pull results into
LOCAL_OUTPUT_DIR="android_output_${ACCELERATOR_NAME}_${PREPROCESSOR_TYPE}_${PHONE_MODEL}"
mkdir -p "./$LOCAL_OUTPUT_DIR"

echo "Pulling 10 run output images from $TARGET_DIR/$OUTPUT_RUN_DIR_ON_DEVICE to ./$LOCAL_OUTPUT_DIR/"

# Pull just the *contents* of the remote dir into our new local dir
# Using "/." at the end of the remote path pulls the contents, not the directory itself
adb pull "$TARGET_DIR/$OUTPUT_RUN_DIR_ON_DEVICE/." "./$LOCAL_OUTPUT_DIR/"

echo "Pulled run images."

if [ "$SAVE_PREPROCESSED" = true ]; then
    echo "Pulling preprocessed_output.png from device..."
    # Pull the preprocessed image into the same results directory for tidiness
    adb pull "$TARGET_DIR/$OUTPUT_PREPROCESSED_IMAGE" "./$LOCAL_OUTPUT_DIR/preprocessed_output.png"
fi

echo "Done. Results are in ./$LOCAL_OUTPUT_DIR/"
# --- End Modification ---