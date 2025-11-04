# Text Enhancer - Dummy Backend
## Overview
This directory contains a "dummy" implementation of the Text Enhancer C API.
Its primary purpose is to serve as a minimal, end-to-end test harness for the API and build system. It allows you to verify the entire flow (build, deployment, dynamic loading, and execution on AndroidXR) without requiring a real model or specific hardware (CPU, GPU, NPU).
The dummy backend does not perform any real image processing. Instead, it generates a predictable 512x512 checkerboard image to confirm that the API was called successfully.
## Files in this Directory
* `deploy_and_run_dummy_on_android.sh`: The main script to copy, and run the solution on a connected Android device.
* `text_enhancer_api.h`: This is the common C API header file. It defines the interface (e.g., TextEnhancer_Initialize) that the standalone calls and the backend library implements.
* `text_enhancer_lib_dummy.so`: The dummy shared library implementation.
* `text_enhancer_standalone_dummy`: The standalone executable.
* `dummy_checkerboard_output.png`: The example output image generated after a successful run.
## How to Use
All commands should be run from the workspace root.


### 1. Run the Deploy Script
With a connected Android device, run the deploy_and_run_dummy_on_android.sh script.
This script handles everything:
- Copies the new binaries from your bazel-out/ build directory (if --copy is used).
- Pushes the binaries to /data/local/tmp/ on the device.
- Executes the standalone.
- Pulls the resulting dummy_checkerboard_output.png back to this directory.
To build, copy, and run:
```sh
./deploy_and_run_dummy_on_android.sh
```


If you omit `--copy`, the script will just re-run whatever binaries are already present in this directory.

### 2. Check the Output
After the script finishes, a new file named dummy_checkerboard_output.png will be created in this directory. Open it to verify it is a 512x512 checkerboard image.

## Implementation Details

* **Build the Binaries**

    First, build the dummy standalone and shared library for AndroidXR using Bazel:
    ```sh
    bazel build //litert/samples/text_enhancer:text_enhancer_standalone_dummy --config=android_arm64
    ```


* **Backend** (`text_enhancer/backends/main_dummy.cc`): This is the source for text_enhancer_lib_dummy.so. It implements all TextEnhancer_... functions with minimal logic. TextEnhancer_PostProcess is the only function with real work: it allocates a buffer and draws the 512x512 checkerboard pattern.

* **Standalone** (`text_enhancer/main_standalone_dummy.cc`): This is the source for text_enhancer_standalone_dummy. It uses dlopen() to load the .so at runtime, loads all API functions using dlsym(), and calls them in the standard order (Initialize, PreProcess, Run, PostProcess, Shutdown).