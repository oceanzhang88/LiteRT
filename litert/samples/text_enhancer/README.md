# Text Enhancer .so Solution for AndroidXR

This project demonstrates how to build and run a "Text Enhancer" (Super Resolution) model using LiteRT on an Android device. It is designed as a command-line tool to be run via Android ADB.

## This project demonstrates:

1.  **Multiple Backends:** Using LiteRT with different accelerators (**CPU**, **GPU**, and **NPU**) from a common codebase.
2.  **C API Interface:** How to wrap C++ LiteRT logic within a clear C API (`text_enhancer_api.h`) for easy integration.
3.  **Dynamic Loading:** A standalone executable (`text_enhancer_standalone_*`) that dynamically loads a backend-specific shared library (`text_enhancer_lib_*.so`) at runtime.
4.  **Vulkan Preprocessing:** Using a Vulkan compute shader for efficient image preprocessing (resizing) on the GPU, which is the default for Android.
5.  **Benchmarking:** The main executable runs the full pipeline (preprocess, inference, postprocess) 10 times and reports min/max/avg timings.
6.  **Deployment:** A single shell script (`deploy_and_run_on_android.sh`) to deploy all necessary assets (executable, libraries, models, images) and run the benchmark.

## Project Architecture

The project is split into two main components:

1.  **Backend Shared Libraries (`text_enhancer_lib_*.so`)**

      * These libraries (e.g., `text_enhancer_lib_gpu.so`, `text_enhancer_lib_npu.so`) contain the core LiteRT logic for a *specific* accelerator.
      * They all implement the common C API defined in `text_enhancer_api.h`.
      * This encapsulates all the LiteRT C++ objects and accelerator-specific setup.

2.  **Standalone Executables (`text_enhancer_standalone_*`)**

      * These are small command-line programs (e.g., `text_enhancer_standalone_gpu`).
      * At runtime, they use `dlopen` to load the corresponding shared library (e.g., `text_enhancer_standalone_gpu` loads `text_enhancer_lib_gpu.so`).
      * They use `dlsym` to get function pointers to the C API (e.g., `TextEnhancer_Initialize`, `TextEnhancer_Run`).
      * All executables are built from the common logic in `main_standalone_common.h`.

## Application Workflow (via `main_standalone_common.h`)

1.  **Parse Arguments:** The executable parses command-line flags (e.g., `--preprocessor`, `--save_preprocessed`).
2.  **Load Library:** Dynamically loads the required `.so` file (passed as an argument).
3.  **Load C API Symbols:** Gets pointers to all `TextEnhancer_*` functions.
4.  **Load Input Image:** Loads the specified input image from a file.
5.  **Create AHB (Android):** On Android, the image data is converted to an `AHardwareBuffer` for efficient processing.
6.  **Initialize Session:** Calls `TextEnhancer_Initialize()` with options (model path, accelerator name, etc.).
7.  **Run Benchmark Loop (10 times):**
      * **Pre-process:** Calls `TextEnhancer_PreProcess_AHB()` (or `TextEnhancer_PreProcess` on desktop). This resizes the image to the model's expected input dimensions.
      * **Inference:** Calls `TextEnhancer_Run()`.
      * **Post-process:** Calls `TextEnhancer_PostProcess()` to get the final, upscaled image buffer.
      * **Save Output:** The high-resolution output buffer is converted to PNG and saved to `output_run_images/output_<N>.png`.
      * **Free Output:** Calls `TextEnhancer_FreeOutputData()`.
8.  **Print Statistics:** Calculates and prints the Min, Max, and Avg timings for preprocessing, inference, and postprocessing over the 10 runs.
9.  **Shutdown:** Calls `TextEnhancer_Shutdown()` to release all resources.

## Prerequisites

1.  **clang or gcc**: Installed.
2.  **Android NDK and SDK**: Installed. (Tested with NDK=25c, SDK=34)
3.  **Bazel**: Installed.
4.  **ADB**: Installed and in PATH.
5.  **LiteRT**: [LiteRT libraries](https://github.com/google-ai-edge/LiteRT).

## Build Instructions

All commands should be run from the root of the LiteRT repository.

1.  **Configure the build tools:**

    ```bash
    ./configure
    bazel sync --configure
    ```

2.  **Build the desired target:**
    You must build both the standalone executable and its corresponding shared library. The executable is defined as `text_enhancer_standalone_<accelerator>` in the `BUILD` file.

      * **For GPU:**
        ```bash
        bazel build //litert/samples/text_enhancer:text_enhancer_standalone_gpu
        ```
      * **For NPU:**
        ```bash
        bazel build //litert/samples/text_enhancer:text_enhancer_standalone_npu
        ```
      * **For CPU:**
        ```bash
        bazel build //litert/samples/text_enhancer:text_enhancer_standalone_cpu
        ```

## Running on Android

The easiest way to run the sample is using the provided deployment script.

1.  **Connect your Android device** and ensure it's recognized by `adb devices`.

2.  **Run the script:**
    The script requires two things: the accelerator you want to run and the path to your `bazel-bin` directory.

    ```bash
    ./litert/samples/text_enhancer/deploy_and_run_on_android.sh [OPTIONS] <path_to_bazel_bin>
    ```

3.  **Script Options:**

      * `--accelerator=cpu|gpu|npu`: (Default: `cpu`) Specifies which backend to run.
      * `--preprocessor=cpu|vulkan`: (Default: `cpu`, but forced to `vulkan` on Android) Specifies which preprocessor to use.
      * `--phone=s25|vst`: (Default: `s25`) Specifies the phone model to select the correct NPU libraries and model.
      * `--save_preprocessed`: (Default: `false`) If set, saves the preprocessed (resized) input image for debugging.

4.  **Example (Running NPU on S25):**

    ```bash
    # From the root of the LiteRT repo
    ./litert/samples/text_enhancer/deploy_and_run_on_android.sh \
        --accelerator=npu \
        --phone=s25 \
        ./bazel-bin
    ```

5.  **What the script does:**

      * Pushes the correct executable (e.g., `text_enhancer_standalone_npu`).
      * Pushes the correct library (e.g., `text_enhancer_lib_npu.so`).
      * Pushes the test image and the correct `.tflite` model.
      * Pushes accelerator-specific libraries (e.g., QNN libraries for NPU).
      * Sets `LD_LIBRARY_PATH` on the device and executes the command.
      * Pulls the results (the 10 output images) back from the device.

6.  **View Results:**
    The script will create a new directory on your host machine named `android_output_<accelerator>_<preprocessor>_<phone>` (e.g., `android_output_npu_vulkan_s25`). This directory will contain the 10 upscaled output images from the benchmark run.

-----