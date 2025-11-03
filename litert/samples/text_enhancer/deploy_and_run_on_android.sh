#!/bin/bash

# Deploys and runs the Text Enhancer sample on a connected Android device.
#
# Prerequisites:
# 1. Android device connected with 'adb'.
# 2. 'bazel build' has been run for the desired target
#    (e.g., :text_enhancer_standalone_cpu).
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
#     --phone=s25|s23            (Default: s23. Used for NPU lib path)
#

set -e

# --- Default Values ---
ACCELERATOR_NAME="cpu"
PREPROCESSOR_TYPE="cpu"
SAVE_PREPROCESSED=false
PHONE_MODEL="s23"
QNN_LIBS_PATH="/data/local/tmp/qnn_libs/s23_8gen2" # Default
DISPATCH_LIB_PATH="/data/local/tmp/qnn_libs" # Default

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
    echo "  --phone=s25|s23            (Default: s23. Used for NPU lib path)"
    echo "  --qnn_libs_path=<path>     (Optional. Overrides default QNN lib path on device)"
    echo "  --dispatch_lib_path=<path> (Optional. Overrides default dispatch lib path on device)"
    echo "  --help                     Show this help message"
}

# --- Argument Parsing ---
for i in "$@"
do
case $i in
    --accelerator=*)
    ACCELERATOR_NAME="${i#*=}"
    shift
    ;;
    --preprocessor=*)
    PREPROCESSOR_TYPE="${i#*=}"
    shift
    ;;
    --save_preprocessed)
    SAVE_PREPROCESSED=true
    shift
    ;;
    --phone=*)
    PHONE_MODEL="${i#*=}"
    shift
    ;;
    --qnn_libs_path=*)
    QNN_LIBS_PATH="${i#*=}"
    shift
    ;;
    --dispatch_lib_path=*)
    DISPATCH_LIB_PATH="${i#*=}"
    shift
    ;;
    --help)
    show_help
    exit 0
    ;;
    -*)
    echo "Error: Unknown option: $i"
    show_help
    exit 1
    ;;
esac
done

BAZEL_BIN_PATH="$1"
if [ -z "$BAZEL_BIN_PATH" ]; then
    echo "Error: Missing <bazel_bin_path> argument."
    show_help
    exit 1
fi

BAZEL_BIN_PATH=$(echo "$BAZEL_BIN_PATH" | sed 's:/*$::')

echo "Selected Accelerator: $ACCELERATOR_NAME"
echo "Selected Pre-processor: $PREPROCESSOR_TYPE"
echo "Save Preprocessed Image: $SAVE_PREPROCESSED"

if [ "$PHONE_MODEL" == "s25" ]; then
    QNN_LIBS_PATH="/data/local/tmp/qnn_libs/s25_8gen4"
    echo "Using QNN libraries path for S25 (8 Gen 4)."
elif [ "$PHONE_MODEL" == "s23" ]; then
    QNN_LIBS_PATH="/data/local/tmp/qnn_libs/s23_8gen2"
    echo "Using QNN libraries path for S23 (8 Gen 2)."
else
    echo "Warning: Unknown phone model '$PHONE_MODEL'. Defaulting to: $QNN_LIBS_PATH"
fi

if [ "$DISPATCH_LIB_PATH" == "/data/local/tmp/qnn_libs" ]; then
    echo "Defaulting to internal dispatch library."
else
    echo "Using custom dispatch library path: $DISPATCH_LIB_PATH"
fi


# --- RENAMED: Define Paths ---
TARGET_DIR="/data/local/tmp/text_enhancer_android"

# Paths relative to bazel-bin
EXECUTABLE_REL_PATH="litert/samples/text_enhancer/text_enhancer_standalone_${ACCELERATOR_NAME}"
LIB_REL_PATH="litert/samples/text_enhancer/text_enhancer_lib_${ACCELERATOR_NAME}.so"
RUNTIME_LIB_REL_PATH="litert/c/libLiteRtRuntimeCApi.so"

# Paths relative to project root (now inside text_enhancer)
SHADER_REL_PATH="litert/samples/text_enhancer/shaders/crop_resize.spv"
IMAGE_REL_PATH="litert/samples/text_enhancer/test_images/low_res_image.png"
MODEL_REL_PATH="litert/samples/text_enhancer/models/super_res-float.tflite"

# GPU-specific assets
GPU_ACCELERATOR_LIB_REL_PATH="litert/gpu/libLiteRtGpuAccelerator.so"

# NPU-specific assets
DISPATCH_LIB_REL_PATH="litert/vendors/qualcomm/dispatch/libdispatch_api.so"

# Basenames for on-device paths
EXECUTABLE_NAME_ON_DEVICE=$(basename "$EXECUTABLE_REL_PATH")
LIB_NAME_ON_DEVICE=$(basename "$LIB_REL_PATH")
RUNTIME_LIB_NAME_ON_DEVICE=$(basename "$RUNTIME_LIB_REL_PATH")
MODEL_BASENAME=$(basename "$MODEL_REL_PATH")
IMAGE_BASENAME=$(basename "$IMAGE_REL_PATH")

OUTPUT_IMAGE="output_image.png"
OUTPUT_PREPROCESSED_IMAGE="preprocessed_output.png"


# --- Start Deployment ---
echo "Starting deployment to Android device..."
echo "Using output path: $BAZEL_BIN_PATH/$EXECUTABLE_REL_PATH"

adb shell "mkdir -p $TARGET_DIR/models"
adb shell "mkdir -p $TARGET_DIR/test_images"
adb shell "mkdir -p $TARGET_DIR/shaders"
adb shell "mkdir -p $TARGET_DIR/npu/dsp"
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

adb push "$MODEL_REL_PATH" "$TARGET_DIR/models/"
echo "Pushed text enhancer model." # <-- Updated log message

adb push "$BAZEL_BIN_PATH/$RUNTIME_LIB_REL_PATH" "$TARGET_DIR/$RUNTIME_LIB_NAME_ON_DEVICE"
echo "Pushed C API shared library."

adb push "$BAZEL_BIN_PATH/$LIB_REL_PATH" "$TARGET_DIR/$LIB_NAME_ON_DEVICE"
echo "Pushed project API library ($BAZEL_BIN_PATH/$LIB_REL_PATH)."

# --- Push Accelerator-Specific Libs ---
LD_PATH="$TARGET_DIR" 

if [ "$ACCELERATOR_NAME" == "gpu" ]; then
    adb push "$BAZEL_BIN_PATH/$GPU_ACCELERATOR_LIB_REL_PATH" "$TARGET_DIR/"
    echo "Pushed GPU accelerator library."
fi

if [ "$ACCELERATOR_NAME" == "npu" ]; then
    adb push "$BAZEL_BIN_PATH/$DISPATCH_LIB_REL_PATH" "$TARGET_DIR/npu/"
    echo "Pushed NPU dispatch library."
    LD_PATH="$QNN_LIBS_PATH:$QNN_LIBS_PATH/dsp:$DISPATCH_LIB_PATH:$TARGET_DIR/npu:$TARGET_DIR"
fi

adb shell "chmod +x $TARGET_DIR/$EXECUTABLE_NAME_ON_DEVICE"
echo "Set execute permissions on device."

echo "Cleaning up previous run results"
adb shell "rm -f $TARGET_DIR/$OUTPUT_IMAGE"
adb shell "rm -f $TARGET_DIR/$OUTPUT_PREPROCESSED_IMAGE"


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

CMD="cd $TARGET_DIR && LD_LIBRARY_PATH=\"$LD_PATH\" ./$EXECUTABLE_NAME_ON_DEVICE \
     ./$LIB_NAME_ON_DEVICE \
     ./models/$MODEL_BASENAME \
     ./test_images/$IMAGE_BASENAME \
     ./$OUTPUT_IMAGE \
     $PREPROCESSOR_FLAG \
     $SHADER_FLAG \
     $SAVE_PREPROCESSED_FLAG \
     --platform=android \
     "

echo ""
echo "Deployment complete."
echo "To run the text enhancer sample on the device, use a command like this:" # <-- Updated log
echo "  adb shell \"$CMD\""

# --- Execute the command ---
adb shell "$CMD"

# --- Pull Results ---
echo ""
echo "To pull the result:"
OUTPUT_PULL_NAME="output_image_${ACCELERATOR_NAME}_${PREPROCESSOR_TYPE}.png"
echo "  adb pull $TARGET_DIR/$OUTPUT_IMAGE ./$OUTPUT_PULL_NAME"
adb pull "$TARGET_DIR/$OUTPUT_IMAGE" "./$OUTPUT_PULL_NAME"

if [ "$SAVE_PREPROCESSED" = true ]; then
    echo "Pulling preprocessed_output.png from device..."
    adb pull "$TARGET_DIR/$OUTPUT_PREPROCESSED_IMAGE" "./preprocessed_output.png"
fi

echo "Done."