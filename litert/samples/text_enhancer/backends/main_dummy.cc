#include <cstring> // For memset
#include <new>

#include "litert/samples/super_resolution/super_res_api.h"

/**
 * @brief A minimal session handle for the dummy implementation.
 * It doesn't need to hold any state for this example.
 */
struct SuperResSession {};

// --- API Function Implementations ---

SuperResSession* SuperRes_Initialize(
    const char* model_path,
    SuperRes_PreprocessorType preprocessor_type,
    const char* passthrough_vert_shader_path,
    const char* compute_shader_path) {
    // No model or preprocessor needed, just return a valid pointer.
    return new SuperResSession;
}

void SuperRes_Shutdown(SuperResSession* session) {
    if (session) {
        delete session;
    }
}

bool SuperRes_PreProcess(SuperResSession* session, const ImageData* input_image) {
    // Do nothing, just succeed.
    return (session != nullptr);
}

#ifdef __ANDROID__
bool SuperRes_PreProcess_AHB(SuperResSession* session,
                             AHardwareBuffer* in_buffer,
                             int in_width,
                             int in_height) {
    // Do nothing, just succeed.
    return (session != nullptr);
}
#endif

const float* SuperRes_GetPreprocessedData(SuperResSession* session,
                                        int* width, int* height, int* channels) {
    // No preprocessed data to return.
    if (width) *width = 0;
    if (height) *height = 0;
    if (channels) *channels = 0;
    return nullptr;
}

bool SuperRes_Run(SuperResSession* session) {
    // No model to run, just succeed.
    return (session != nullptr);
}

bool SuperRes_PostProcess(SuperResSession* session, OutputData* output_data) {
    if (!session || !output_data) {
        return false;
    }

    // --- Checkerboard Generation ---
    const int kWidth = 256;
    const int kHeight = 256;
    const int kChannels = 4; // RGBA
    const int kTileSize = 16;

    float* buffer = new (std::nothrow) float[kWidth * kHeight * kChannels];
    if (!buffer) {
        return false;
    }

    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            // Determine if the tile is "black" or "white"
            // Using XOR for a clean checkerboard pattern
            bool is_black_tile = ((x / kTileSize) % 2) ^ ((y / kTileSize) % 2);
            float value = is_black_tile ? 0.0f : 1.0f;

            int index = (y * kWidth + x) * kChannels;
            buffer[index + 0] = value; // Red
            buffer[index + 1] = value; // Green
            buffer[index + 2] = value; // Blue
            buffer[index + 3] = 1.0f;  // Alpha
        }
    }
    // -------------------------------

    // Assign the newly created buffer to the output struct
    output_data->data = buffer;
    output_data->width = kWidth;
    output_data->height = kHeight;
    output_data->channels = kChannels;

    return true;
}

void SuperRes_FreeOutputData(OutputData* output_data) {
    if (output_data && output_data->data) {
        delete[] output_data->data;
        output_data->data = nullptr;
    }
}