#ifndef SUPER_RESOLUTION_API_H_
#define SUPER_RESOLUTION_API_H_

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the super resolution session
typedef struct SuperResSession SuperResSession;

// Struct for passing input image data
typedef struct {
  unsigned char* data;
  int width;
  int height;
  int channels;
} ImageData;

// Struct for getting output data.
// The 'data' buffer is allocated by the library and must be
// freed by the caller using SuperRes_FreeOutputData.
typedef struct {
  float* data;  // The model outputs float
  int width;
  int height;
  int channels;
} OutputData;

/**
 * @brief Initializes the super resolution session.
 * @param model_path Path to the .tflite model file.
 * @param passthrough_vert_shader_path Path to the passthrough vertex shader.
 * @param super_res_compute_shader_path Path to the super res compute shader.
 * @param use_gl_buffers Whether to use GL buffers for GPU acceleration.
 * @return An opaque handle (SuperResSession*) or NULL on failure.
 */
SuperResSession* SuperRes_Initialize(
    const char* model_path, const char* passthrough_vert_shader_path,
    const char* super_res_compute_shader_path, bool use_gl_buffers);

/**
 * @brief Shuts down the session and frees all associated resources.
 * @param session The handle returned by SuperRes_Initialize.
 */
void SuperRes_Shutdown(SuperResSession* session);

/**
 * @brief Pre-processes an input image and prepares it for inference.
 * @param session The session handle.
 * @param input_image The raw input image data.
 * @return true on success, false on failure.
 */
bool SuperRes_PreProcess(SuperResSession* session, const ImageData* input_image);

/**
 * @brief Runs the super resolution inference.
 * Must be called after a successful SuperRes_PreProcess.
 * @param session The session handle.
 * @return true on success, false on failure.
 */
bool SuperRes_Run(SuperResSession* session);

/**
 * @brief Retrieves the inference result.
 * @param session The session handle.
 * @param output_data A pointer to an OutputData struct that will be filled.
 * The 'data' buffer is allocated by this function.
 * @return true on success, false on failure.
 */
bool SuperRes_PostProcess(SuperResSession* session, OutputData* output_data);

/**
 * @brief Frees the data buffer allocated by SuperRes_PostProcess.
 * @param output_data The struct containing the data to free.
 */
void SuperRes_FreeOutputData(OutputData* output_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SUPER_RESOLUTION_API_H_