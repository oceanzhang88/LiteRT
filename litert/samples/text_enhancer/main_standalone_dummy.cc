#include <dlfcn.h>  // --- Include for dynamic loading ---
#include <iostream>
#include <string>
#include <vector>
#include <algorithm> // For std::min/max

// --- RENAMED: Include new API header ---
#include "text_enhancer_api.h"
#include "utils/image_utils.h"

// --- Define function pointer types based on TextEnhancerSession ---
typedef TextEnhancerSession* (*t_TextEnhancer_Initialize)(const TextEnhancerOptions& options);
typedef void (*t_TextEnhancer_Shutdown)(TextEnhancerSession* session);
#ifdef __ANDROID__
typedef TextEnhancerStatus (*t_TextEnhancer_PreProcess_AHB)(TextEnhancerSession* session,
                                                            AHardwareBuffer* buffer);
#endif
typedef TextEnhancerStatus (*t_TextEnhancer_GetPreprocessedData)(TextEnhancerSession* session,
                                                                 uint8_t** data);
typedef TextEnhancerStatus (*t_TextEnhancer_PreProcess)(TextEnhancerSession* session,
                                                        const uint8_t* rgb_data);
typedef TextEnhancerStatus (*t_TextEnhancer_Run)(TextEnhancerSession* session,
                                                 float* inference_time_ms);
typedef TextEnhancerStatus (*t_TextEnhancer_PostProcess)(TextEnhancerSession* session,
                                                         TextEnhancerOutput& output);
typedef void (*t_TextEnhancer_FreeOutputData)(TextEnhancerOutput& output);
// ----------------------------------------------------------------------

// --- Global function pointers ---
t_TextEnhancer_Initialize fn_TextEnhancer_Initialize = nullptr;
t_TextEnhancer_Shutdown fn_TextEnhancer_Shutdown = nullptr;
#ifdef __ANDROID__
t_TextEnhancer_PreProcess_AHB fn_TextEnhancer_PreProcess_AHB = nullptr;
#endif
t_TextEnhancer_GetPreprocessedData fn_TextEnhancer_GetPreprocessedData = nullptr;
t_TextEnhancer_PreProcess fn_TextEnhancer_PreProcess = nullptr;
t_TextEnhancer_Run fn_TextEnhancer_Run = nullptr;
t_TextEnhancer_PostProcess fn_TextEnhancer_PostProcess = nullptr;
t_TextEnhancer_FreeOutputData fn_TextEnhancer_FreeOutputData = nullptr;
// -------------------------------------------------------------------

int main(int argc, char** argv) {
    std::cout << "--- Running Text Enhancer Standalone (Dummy) ---" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <lib_path.so>" << std::endl;
        return 1;
    }

    const char* lib_path = argv[1];
    const char* output_path = "dummy_checkerboard_output.png";

    // --- Load dynamic library ---
    void* handle = dlopen(lib_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load library " << lib_path << ": " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "Loaded library: " << lib_path << std::endl;

    // --- Load symbols ---
#define LOAD_SYMBOL(name)                                                               \
    fn_##name = (t_##name)dlsym(handle, #name);                                         \
    if (!fn_##name) {                                                                   \
        std::cerr << "Failed to load symbol: " #name << ": " << dlerror() << std::endl; \
        dlclose(handle);                                                                \
        return 1;                                                                       \
    }

    LOAD_SYMBOL(TextEnhancer_Initialize);
    LOAD_SYMBOL(TextEnhancer_Shutdown);
#ifdef __ANDROID__
    LOAD_SYMBOL(TextEnhancer_PreProcess_AHB);
#endif
    LOAD_SYMBOL(TextEnhancer_GetPreprocessedData);
    LOAD_SYMBOL(TextEnhancer_PreProcess);
    LOAD_SYMBOL(TextEnhancer_Run);
    LOAD_SYMBOL(TextEnhancer_PostProcess);
    LOAD_SYMBOL(TextEnhancer_FreeOutputData);

    std::cout << "All symbols loaded." << std::endl;
    // --- END: Load dynamic library ---


    // 1. Initialize the session.
    std::cout << "Initializing dummy session..." << std::endl;
    TextEnhancerOptions options = {};
    options.model_path = "dummy_model.tflite";
    options.compute_shader_path = "";
    options.accelerator_name = "dummy";
    options.input_width = 640;  // Dummy value
    options.input_height = 480; // Dummy value

    TextEnhancerSession* session = fn_TextEnhancer_Initialize(options);

    if (!session) {
        std::cerr << "Failed to initialize TextEnhancerSession." << std::endl;
        dlclose(handle);
        return -1;
    }

    // 2. Pre-processing
    std::cout << "Calling dummy pre-process..." << std::endl;
    if (fn_TextEnhancer_PreProcess(session, nullptr) != kTextEnhancerOk) {
        std::cerr << "Dummy pre-process call failed." << std::endl;
        fn_TextEnhancer_Shutdown(session);
        dlclose(handle);
        return -1;
    }

    // 3. Run inference
    std::cout << "Calling dummy run..." << std::endl;
    float dummy_time = 0.0f;
    if (fn_TextEnhancer_Run(session, &dummy_time) != kTextEnhancerOk) {
        std::cerr << "Dummy run call failed." << std::endl;
        fn_TextEnhancer_Shutdown(session);
        dlclose(handle);
        return -1;
    }
    std::cout << "Dummy inference time: " << dummy_time << " ms" << std::endl;


    // 4. Post-process to get the checkerboard
    std::cout << "Calling dummy post-process..." << std::endl;
    TextEnhancerOutput output_data = {};
    if (fn_TextEnhancer_PostProcess(session, output_data) != kTextEnhancerOk) {
        std::cerr << "Dummy post-process call failed." << std::endl;
        fn_TextEnhancer_Shutdown(session);
        dlclose(handle);
        return -1;
    }

    std::cout << "Got output data: "
              << output_data.width << "x"
              << output_data.height << "x"
              << output_data.channels << std::endl;

    // --- Cast output data back to float* for processing ---
    float* output_data_float = reinterpret_cast<float*>(output_data.data);

    // Convert float output (e.g., RGBA) to unsigned char (RGB) for saving
    std::vector<unsigned char> output_image_bytes(output_data.width * output_data.height * 3); // Save as 3-channel RGB
    for (int y = 0; y < output_data.height; ++y) {
        for (int x = 0; x < output_data.width; ++x) {
            int in_idx = (y * output_data.width + x) * output_data.channels;
            int out_idx = (y * output_data.width + x) * 3;
            
            output_image_bytes[out_idx + 0] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, output_data_float[in_idx + 0] * 255.0f)));
            output_image_bytes[out_idx + 1] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, output_data_float[in_idx + 1] * 255.0f)));
            output_image_bytes[out_idx + 2] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, output_data_float[in_idx + 2] * 255.0f)));
        }
    }

    // 5. Save the output
    std::cout << "Saving output to " << output_path << "..." << std::endl;
    if (!ImageUtils::SaveImage(output_path, output_data.width, output_data.height,
                          3, output_image_bytes.data())) {
        std::cerr << "Failed to save output image." << std::endl;
    } else {
        std::cout << "Successfully saved dummy output!" << std::endl;
    }

    // 6. Clean up
    fn_TextEnhancer_FreeOutputData(output_data);
    fn_TextEnhancer_Shutdown(session);
    dlclose(handle);

    std::cout << "--------------------------------------------" << std::endl;
    return 0;
}