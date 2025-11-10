#pragma once

#include <stdint.h>

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle for the Text Enhancer instance.
 */
typedef struct TextEnhancerSession TextEnhancerSession;

/**
 * Status codes for Text Enhancer API functions.
 */
typedef enum {
    kTextEnhancerOk = 0,
    kTextEnhancerFailed = 1,
    kTextEnhancerModelLoadFailed = 2,
    kTextEnhancerInputError = 3,
    kTextEnhancerRuntimeError = 4,
} TextEnhancerStatus;

/**
 * Options for initializing the Text Enhancer instance.
 */
typedef struct {
    const char* model_path;
    const char* compute_shader_path;  // Optional, for Vulkan
    const char* accelerator_name;     // "cpu", "gpu", "npu", "dummy"
    int input_width;                  // Required input width
    int input_height;                 // Required input height
    bool use_int8_preprocessor = false; // Default to float
} TextEnhancerOptions;

/**
 * Represents the output data from the text enhancement process.
 */
typedef struct {
    // --- Fields for raw CPU buffer output ---
    uint8_t* data;
    int width;
    int height;
    int channels;

    // --- Field for AHardwareBuffer output ---
#ifdef __ANDROID__
    AHardwareBuffer* output_buffer;
#endif

} TextEnhancerOutput;

/**
 * Initializes the Text Enhancer instance.
 *
 * @param options Configuration options.
 * @return A session handle, or NULL on failure.
 */
TextEnhancerSession* TextEnhancer_Initialize(const TextEnhancerOptions& options);

/**
 * Pre-processes an AHardwareBuffer input (Android only).
 *
 * @param session The instance session.
 * @param buffer The input AHardwareBuffer.
 * @return kTextEnhancerOk on success.
 */
#ifdef __ANDROID__
TextEnhancerStatus TextEnhancer_PreProcess_AHB(TextEnhancerSession* session, AHardwareBuffer* buffer);
#endif

/**
 * Gets a pointer to the pre-processed (input) data buffer.
 *
 * @param session The instance session.
 * @param data A pointer to hold the buffer address.
 * @return kTextEnhancerOk on success.
 */
TextEnhancerStatus TextEnhancer_GetPreprocessedData(TextEnhancerSession* session, uint8_t** data);

/**
 * Pre-processes a raw RGB image buffer.
 *
 * @param session The instance session.
 * @param rgb_data The input raw RGB data.
 * @return kTextEnhancerOk on success.
 */
TextEnhancerStatus TextEnhancer_PreProcess(TextEnhancerSession* session, const uint8_t* rgb_data);

/**
 * Runs the text enhancement model.
 *
 * @param session The instance session.
 * @param inference_time_ms Optional output for inference time.
 * @return kTextEnhancerOk on success.
 */
TextEnhancerStatus TextEnhancer_Run(TextEnhancerSession* session, float* inference_time_ms);

/**
 * Post-processes the model output into a displayable image.
 *
 * @param session The instance session.
 * @param output A struct to be filled with the output image data and dims.
 * On Android, this may fill 'output_buffer'.
 * On other platforms, it will fill 'data', 'width', 'height'.
 * @return kTextEnhancerOk on success.
 */
TextEnhancerStatus TextEnhancer_PostProcess(TextEnhancerSession* session, TextEnhancerOutput& output);

/**
 * Frees the data buffer allocated within the TextEnhancerOutput struct.
 * If output_buffer is non-NULL, it will be released.
 *
 * @param output The output struct whose data will be freed.
 */
void TextEnhancer_FreeOutputData(TextEnhancerOutput& output);

/**
 * Shuts down the Text Enhancer instance and frees all resources.
 *
 * @param session The instance session to be freed.
 */
void TextEnhancer_Shutdown(TextEnhancerSession* session);

#ifdef __cplusplus
}  // extern "C"
#endif