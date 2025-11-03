# LiteRT Android GPU Super Resolution Example

## This project demonstrates:

1.  Preprocessing an input image: resizing it to the model's expected input size.
2.  Performing super-resolution on the preprocessed image using the async API of LiteRT.
3.  Saving the upscaled output image.

This is a command-line tool designed to be run via Android ADB. The C++ code is organized into `ImageUtils` and `ImageProcessor` classes.

## Image Processing Workflow:

1.  **Load Input Image:** Load a low-resolution input image from a file.
2.  **Create OpenGL Texture:** Create an OpenGL texture from the loaded image data.
3.  **Preprocessing:**
    * The input image texture is passed to `ImageProcessor::preprocessInputForSuperResolution`.
    * This step resizes the image to the model's expected input dimensions (e.g., 256x256).
4.  **Super Resolution:**
    * The preprocessed image buffer is passed to the super-resolution model.
    * The model is executed on the GPU.
    * The model outputs a high-resolution image buffer.
5.  **Save Output Image:**
    * The high-resolution output buffer is read back from the GPU.
    * The final upscaled image is saved to a file.

## Prerequisites

1.  **clang or gcc**: Installed.
2.  **Android NDK and SDK**: Installed. (Tested with NDK=25c, SDK=34)
3.  **Bazel**: Installed.
4.  **ADB**: Installed and in PATH.
5.  **LiteRT**: [LiteRT libraries](https://github.com/google-ai-edge/LiteRT).

### Build Instructions

All commands should be run from the root of the LiteRT repository.

Configure the build tools:
```bash
./configure

bazel sync --configure