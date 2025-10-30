#ifndef SUPER_RESOLUTION_SUPER_RES_API_H_
#define SUPER_RESOLUTION_SUPER_RES_API_H_

#ifdef __ANDROID__
// Forward-declare AHardwareBuffer
struct AHardwareBuffer;
#endif

// --- FIX: Add C++ guards ---
// This tells C++ compilers that the functions below use C-style linkage
#ifdef __cplusplus
extern "C" {
#endif

/** @brief An opaque handle to the TFLite Super-Resolution session. */
typedef struct SuperResSession SuperResSession;

/** @brief Holds input image data from a CPU buffer. */
typedef struct {
    /** @brief Pointer to the raw image data (e.g., from stbi_load). */
    unsigned char* data;
    /** @brief Width of the image in pixels. */
    int width;
    /** @brief Height of the image in pixels. */
    int height;
    /** @brief Number of channels (must be 4 for RGBA). */
    int channels;
} ImageData;

/** @brief Holds the output tensor data after post-processing. */
typedef struct {
    /** @brief Pointer to the output float data (owned by the session). */
    float* data;
    /** @brief Width of the output image. */
    int width;
    /** @brief Height of the output image. */
    int height;
    /** @brief Number of output channels (e.g., 4 for RGBA). */
    int channels;
} OutputData;

/** @brief Selects the image pre-processing implementation. */
typedef enum {
    /** @brief Use the CPU (Eigen) implementation for crop/resize. */
    kSuperResCpuPreprocessor,
    /** @brief Use the Vulkan compute shader implementation for crop/resize. */
    kSuperResVulkanPreprocessor
} SuperRes_PreprocessorType;


// --- API Functions ---

/**
 * @brief Initializes a new super-resolution session.
 *
 * @param model_path Path to the .tflite model file.
 * @param preprocessor_type The type of preprocessor to use (CPU or Vulkan).
 * @param passthrough_vert_shader_path (Not used by this backend, can be "").
 * @param compute_shader_path Path to the crop_resize.spv SPIR-V shader.
 * (Required only if preprocessor_type is Vulkan).
 * @return A pointer to a new SuperResSession, or NULL on failure.
 */
SuperResSession* SuperRes_Initialize(
    const char* model_path,
    SuperRes_PreprocessorType preprocessor_type,
    const char* passthrough_vert_shader_path,
    const char* compute_shader_path
);

/**
 * @brief Pre-processes an image from a CPU buffer.
 *
 * This path uses a staging buffer to upload the image to the GPU
 * if the Vulkan preprocessor is selected.
 *
 * @param session A valid session handle from SuperRes_Initialize.
 * @param input_image A pointer to an ImageData struct with input data.
 * @return true on success, false on failure.
 */
bool SuperRes_PreProcess(SuperResSession* session, const ImageData* input_image);

#ifdef __ANDROID__
/**
 * @brief Pre-processes an image from an AHardwareBuffer (Android only).
 *
 * This path attempts a zero-copy import of the AHB into Vulkan
 * if the Vulkan preprocessor is selected.
 *
 * @param session A valid session handle from SuperRes_Initialize.
 * @param in_buffer A handle to the input AHardwareBuffer.
 * @param in_width The width of the input buffer.
 * @param in_height The height of the input buffer.
 * @return true on success, false on failure.
 */
bool SuperRes_PreProcess_AHB(SuperResSession* session,
                         AHardwareBuffer* in_buffer,
                         int in_width,
                         int in_height);
#endif

// --- NEWLY ADDED FUNCTION ---
/**
 * @brief Gets a pointer to the *internal* pre-processed float buffer.
 *
 * Data is valid until the next call to SuperRes_PreProcess or SuperRes_Shutdown.
 * The session retains ownership of the data. DO NOT free it.
 *
 * @param session A valid session handle.
 * @param width (out) Width of the pre-processed buffer.
 * @param height (out) Height of the pre-processed buffer.
 * @param channels (out) Channels of the pre-processed buffer.
 * @return A const float* to the data, or nullptr on failure.
 */
const float* SuperRes_GetPreprocessedData(SuperResSession* session, 
                                        int* width, int* height, int* channels);
// ----------------------------

/**
 * @brief Runs the TFLite model inference.
 *
 * Pre-processing must be completed successfully before calling this.
 *
 * @param session A valid session handle.
 * @return true on success, false on failure.
 */
bool SuperRes_Run(SuperResSession* session);

/**
 * @brief Gets the inference output data.
 *
 * Inference must be completed successfully before calling this.
 *
 * @param session A valid session handle.
 * @param output_data A pointer to an OutputData struct to be filled.
 * The 'data' pointer inside is owned by the session
 * and must be freed via SuperRes_FreeOutputData.
 * @return true on success, false on failure.
 */
bool SuperRes_PostProcess(SuperResSession* session, OutputData* output_data);

/**
 * @brief Frees the data buffer allocated by SuperRes_PostProcess.
 *
 * @param output_data A pointer to the OutputData struct that was filled.
 */
void SuperRes_FreeOutputData(OutputData* output_data);

/**
 * @brief Shuts down the session and releases all resources.
 *
 * @param session The session handle to destroy.
 */
void SuperRes_Shutdown(SuperResSession* session);

// --- FIX: Close the C++ guard ---
#ifdef __cplusplus
}
#endif

#endif // SUPER_RESOLUTION_SUPER_RES_API_H_