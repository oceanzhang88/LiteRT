#include <cstring> // For memset
#include <new>

// --- RENAMED: Include new API header ---
#include "litert/samples/text_enhancer/text_enhancer_api.h"

#ifdef __ANDROID__
#include "android/hardware_buffer.h"
#endif

/**
 * @brief A minimal session handle for the dummy implementation.
 * It doesn't need to hold any state for this example.
 */
// --- RENAMED: Struct ---
struct TextEnhancerSession {};

// --- API Function Implementations ---

// --- RENAMED: Function ---
TextEnhancerSession* TextEnhancer_Initialize(
    const TextEnhancerOptions& options) {
    // Options are ignored by the dummy implementation.
    // No model or preprocessor needed, just return a valid pointer.
    return new TextEnhancerSession;
}

// --- RENAMED: Function ---
void TextEnhancer_Shutdown(TextEnhancerSession* session) {
    if (session) {
        delete session;
    }
}

// --- RENAMED: Function ---
TextEnhancerStatus TextEnhancer_PreProcess(TextEnhancerSession* session,
                                           const uint8_t* rgb_data) {
    // Do nothing, just succeed.
    if (!session) return kTextEnhancerInputError;
    return kTextEnhancerOk;
}

#ifdef __ANDROID__
// --- RENAMED: Function ---
TextEnhancerStatus TextEnhancer_PreProcess_AHB(TextEnhancerSession* session,
                                               AHardwareBuffer* in_buffer) {
    // Do nothing, just succeed.
    if (!session) return kTextEnhancerInputError;
    return kTextEnhancerOk;
}
#endif

// --- RENAMED: Function ---
TextEnhancerStatus TextEnhancer_GetPreprocessedData(TextEnhancerSession* session,
                                                    uint8_t** data) {
    // No preprocessed data to return.
    if (!session || !data) return kTextEnhancerInputError;
    *data = nullptr;
    // Return failure as we don't have data to provide
    return kTextEnhancerFailed; 
}

// --- RENAMED: Function ---
TextEnhancerStatus TextEnhancer_Run(TextEnhancerSession* session,
                                    float* inference_time_ms) {
    // No model to run, just succeed.
    if (!session) return kTextEnhancerInputError;
    if (inference_time_ms) {
        *inference_time_ms = 0.1f; // Report a tiny dummy time
    }
    return kTextEnhancerOk;
}

// --- RENAMED: Function ---
TextEnhancerStatus TextEnhancer_PostProcess(TextEnhancerSession* session,
                                            TextEnhancerOutput& output) {
    if (!session) {
        return kTextEnhancerInputError;
    }

    // --- Checkerboard Generation ---
    // --- UPDATED: Changed size from 256 to 512 ---
    const int kWidth = 512;
    const int kHeight = 512;
    // ---------------------------------------------
    const int kChannels = 4; // RGBA
    const int kTileSize = 16;

    float* buffer = new (std::nothrow) float[kWidth * kHeight * kChannels];
    if (!buffer) {
        return kTextEnhancerFailed;
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

    // --- RENAMED: Assign to new output struct ---
    // Note: We must cast the float* to uint8_t* for the API
    output.data = reinterpret_cast<uint8_t*>(buffer);
    output.width = kWidth;
    output.height = kHeight;
    output.channels = kChannels;

    return kTextEnhancerOk;
}

// --- RENAMED: Function ---
void TextEnhancer_FreeOutputData(TextEnhancerOutput& output) {
    if (output.data) {
        // Cast back to float* to correctly delete the array
        delete[] reinterpret_cast<float*>(output.data);
        output.data = nullptr;
        output.width = 0;
        output.height = 0;
        output.channels = 0;
    }
}