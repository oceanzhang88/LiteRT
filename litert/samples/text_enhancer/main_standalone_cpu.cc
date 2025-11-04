#include <dlfcn.h>  // --- Include for dynamic loading ---

#include <algorithm>  // For std::min/max
#include <chrono>     // --- Include for timing ---
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// --- RENAMED: Include new API header ---
#include "text_enhancer_api.h"
#include "utils/image_utils.h"

// Include AHB headers if compiling for Android
#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#endif

// --- RENAMED: Define function pointer types based on TextEnhancerSession ---
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

// --- RENAMED: Global function pointers ---
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

// Helper to parse command-line arguments
std::string GetFlagValue(int argc, char** argv, const std::string& flag,
                         const std::string& default_value) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind(flag, 0) == 0) {
            return arg.substr(flag.length());
        }
    }
    return default_value;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <lib_path.so> <model_path> <input_image> <output_image>"
                  << " [--preprocessor=cpu|vulkan]"
                  << " [--shader_path=path/to/shader]"
                  << " [--platform=desktop|android]"
                  << " [--save_preprocessed=true|false]" << std::endl;
        return 1;
    }

    const char* lib_path = argv[1];
    const char* model_path = argv[2];
    const char* input_image_path = argv[3];
    const char* output_image_path = argv[4];

    void* handle = dlopen(lib_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load library " << lib_path << ": " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "Loaded library: " << lib_path << std::endl;

// --- RENAMED: Load symbols ---
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

    // --- Flag Parsing ---
    std::string platform_str = GetFlagValue(argc, argv, "--platform=", "android");
    std::string preprocessor_type_str = GetFlagValue(argc, argv, "--preprocessor=", "vulkan");
    std::string save_preprocessed_str = GetFlagValue(argc, argv, "--save_preprocessed=", "false");
    bool save_preprocessed = (save_preprocessed_str == "true");

    std::string compute_shader_path_str = "";
    const char* compute_shader_path = "";

#ifdef __ANDROID__
    if (platform_str == "android") {
        preprocessor_type_str = "vulkan";
        std::cout << "Running on 'android' platform. Defaulting to Vulkan preprocessor."
                  << std::endl;
    }
#else
    if (platform_str == "android") {
        std::cerr << "Error: --platform=android can only be used when compiled for Android."
                  << std::endl;
        dlclose(handle);
        return 1;
    }
#endif

    if (platform_str == "desktop") {
        if (preprocessor_type_str == "vulkan") {
            std::cout << "Using Vulkan Pre-processor (Staging Buffer Path)" << std::endl;
        } else {
            std::cout << "Using CPU Pre-processor" << std::endl;
        }
    }

    if (preprocessor_type_str == "vulkan") {
        compute_shader_path_str =
            GetFlagValue(argc, argv, "--shader_path=", "shaders/crop_resize.spv");
        compute_shader_path = compute_shader_path_str.c_str();
        std::cout << "[Debug main] compute_shader_path set to: '" << compute_shader_path << "'"
                  << std::endl;
    } else {
        std::cout << "[Debug main] compute_shader_path set to: '' (empty)" << std::endl;
    }

    // --- Load Image ---
    int img_width, img_height, img_channels;
    unsigned char* image_data_ptr =
        ImageUtils::LoadImage(input_image_path, img_width, img_height, img_channels, 4);
    if (!image_data_ptr) {
        std::cerr << "Failed to load image: " << input_image_path << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "Loaded input image: " << img_width << "x" << img_height << "x" << img_channels
              << std::endl;

    // --- Initialize the TextEnhancer session ---
    std::cout << "[Debug main] Calling TextEnhancer_Initialize..." << std::endl;

    // --- RENAMED: Struct types ---
    TextEnhancerOptions options = {};
    options.model_path = model_path;
    options.compute_shader_path = compute_shader_path;
    options.accelerator_name = "cpu";
    options.input_width = img_width;
    options.input_height = img_height;

    TextEnhancerSession* session = fn_TextEnhancer_Initialize(options);
    // ----------------------------

    if (!session) {
        std::cerr << "Failed to initialize TextEnhancer session." << std::endl;
        ImageUtils::FreeImageData(image_data_ptr);
        dlclose(handle);
        return 1;
    }
    std::cout << "TextEnhancer session initialized." << std::endl;

    // --- Pre-process ---
    auto start_preprocess = std::chrono::high_resolution_clock::now();

#ifdef __ANDROID__
    AHardwareBuffer* ahb_handle = nullptr;
    if (platform_str == "android") {
        std::cout << "Converting loaded image to AHardwareBuffer..." << std::endl;
        ahb_handle = ImageUtils::CreateAhbFromImageData(image_data_ptr, img_width, img_height);

        ImageUtils::FreeImageData(image_data_ptr);
        image_data_ptr = nullptr;

        if (!ahb_handle) {
            std::cerr << "Failed to create AHardwareBuffer from image data." << std::endl;
            fn_TextEnhancer_Shutdown(session);
            dlclose(handle);
            return 1;
        }

        std::cout << "Pre-processing with AHardwareBuffer..." << std::endl;

        // --- RENAMED: API calls ---
        if (fn_TextEnhancer_PreProcess_AHB(session, ahb_handle) != kTextEnhancerOk) {
            std::cerr << "Pre-processing (AHB) failed." << std::endl;
            ImageUtils::FreeAhb(ahb_handle);
            fn_TextEnhancer_Shutdown(session);
            dlclose(handle);
            return 1;
        }
        std::cout << "Pre-processing (AHB) complete." << std::endl;

        if (save_preprocessed) {
            std::cout << "Saving pre-processed image for verification..." << std::endl;

            uint8_t* pre_data = nullptr;
            TextEnhancerStatus status = fn_TextEnhancer_GetPreprocessedData(session, &pre_data);

            if (status == kTextEnhancerOk && pre_data) {
                // ... (rest of the save logic is unchanged) ...
                int pre_width = options.input_width;
                int pre_height = options.input_height;
                int pre_channels = 4;
                std::string pre_output_path = "preprocessed_output.png";
                ImageUtils::SaveImage(pre_output_path, pre_width, pre_height, pre_channels,
                                      pre_data);
                std::cout << "Pre-processed image saved to " << pre_output_path << std::endl;
            } else {
                std::cerr << "Failed to get pre-processed data for saving." << std::endl;
            }
        }
    } else
#endif
    {
        // --- Desktop (CPU or Vulkan Staging) Path ---
        std::cout << "Pre-processing with CPU buffer..." << std::endl;
        // --- RENAMED: API calls ---
        if (fn_TextEnhancer_PreProcess(session, image_data_ptr) != kTextEnhancerOk) {
            std::cerr << "Pre-processing failed." << std::endl;
            ImageUtils::FreeImageData(image_data_ptr);
            fn_TextEnhancer_Shutdown(session);
            dlclose(handle);
            return 1;
        }
        ImageUtils::FreeImageData(image_data_ptr);
        image_data_ptr = nullptr;
        std::cout << "Pre-processing complete." << std::endl;

        if (save_preprocessed) {
            std::cout << "Saving pre-processed image for verification..." << std::endl;

            uint8_t* pre_data = nullptr;
            TextEnhancerStatus status = fn_TextEnhancer_GetPreprocessedData(session, &pre_data);

            if (status == kTextEnhancerOk && pre_data) {
                // ... (rest of the save logic is unchanged) ...
                int pre_width = options.input_width;
                int pre_height = options.input_height;
                int pre_channels = 4;
                std::string pre_output_path = "preprocessed_output.png";
                ImageUtils::SaveImage(pre_output_path, pre_width, pre_height, pre_channels,
                                      pre_data);
                std::cout << "Pre-processed image saved to " << pre_output_path << std::endl;
            } else {
                std::cerr << "Failed to get pre-processed data for saving." << std::endl;
            }
        }
    }

    auto end_preprocess = std::chrono::high_resolution_clock::now();

    // --- Run Inference ---
    auto start_run = std::chrono::high_resolution_clock::now();

    float inference_time_ms = 0.0f;
    // --- RENAMED: API calls ---
    if (fn_TextEnhancer_Run(session, &inference_time_ms) != kTextEnhancerOk) {
        std::cerr << "Inference run failed." << std::endl;
#ifdef __ANDROID__
        if (ahb_handle) ImageUtils::FreeAhb(ahb_handle);
#endif
        fn_TextEnhancer_Shutdown(session);
        dlclose(handle);
        return 1;
    }

    auto end_run = std::chrono::high_resolution_clock::now();
    std::cout << "Inference complete." << std::endl;

    // --- Post-process & Save ---
    auto start_postprocess = std::chrono::high_resolution_clock::now();

    // --- RENAMED: Struct types ---
    TextEnhancerOutput output_data = {0};
    if (fn_TextEnhancer_PostProcess(session, output_data) != kTextEnhancerOk) {
        std::cerr << "Post-processing failed." << std::endl;
#ifdef __ANDROID__
        if (ahb_handle) ImageUtils::FreeAhb(ahb_handle);
#endif
        fn_TextEnhancer_Shutdown(session);
        dlclose(handle);
        return 1;
    }

    auto end_postprocess = std::chrono::high_resolution_clock::now();
    std::cout << "Output received: " << output_data.width << "x" << output_data.height << std::endl;

    // --- E2E Timings ---
    double preprocess_ms =
        std::chrono::duration<double, std::milli>(end_preprocess - start_preprocess).count();
    double run_ms = inference_time_ms;
    double postprocess_ms =
        std::chrono::duration<double, std::milli>(end_postprocess - start_postprocess).count();
    double total_e2e_ms = preprocess_ms + run_ms + postprocess_ms;

    std::cout << "\n--- End-to-End Timing Summary ---" << std::endl;
    std::cout << "Pre-processing Time: " << preprocess_ms << " ms" << std::endl;
    std::cout << "Inference Time (TextEnhancer_Run): " << run_ms << " ms" << std::endl;
    std::cout << "Post-processing Time: " << postprocess_ms << " ms" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Total E2E Time (Pre + Run + Post): " << total_e2e_ms << " ms" << std::endl;
    std::cout << "-----------------------------------\n" << std::endl;

    // --- Output Conversion Logic (unchanged) ---
    float* output_data_float = reinterpret_cast<float*>(output_data.data);

    std::vector<unsigned char> output_image_bytes(output_data.width * output_data.height * 3);
    for (int y = 0; y < output_data.height; ++y) {
        for (int x = 0; x < output_data.width; ++x) {
            int in_idx = (y * output_data.width + x) * output_data.channels;
            int out_idx = (y * output_data.width + x) * 3;

            output_image_bytes[out_idx + 0] = static_cast<unsigned char>(
                std::max(0.0f, std::min(255.0f, output_data_float[in_idx + 0] * 255.0f)));
            output_image_bytes[out_idx + 1] = static_cast<unsigned char>(
                std::max(0.0f, std::min(255.0f, output_data_float[in_idx + 1] * 255.0f)));
            output_image_bytes[out_idx + 2] = static_cast<unsigned char>(
                std::max(0.0f, std::min(255.0f, output_data_float[in_idx + 2] * 255.0f)));
        }
    }

    ImageUtils::SaveImage(output_image_path, output_data.width, output_data.height, 3,
                          output_image_bytes.data());

    std::cout << "Output image saved to " << output_image_path << std::endl;

    // --- Cleanup ---
#ifdef __ANDROID__
    if (ahb_handle) ImageUtils::FreeAhb(ahb_handle);
#endif
    if (image_data_ptr) ImageUtils::FreeImageData(image_data_ptr);

    // --- RENAMED: API calls ---
    fn_TextEnhancer_FreeOutputData(output_data);
    fn_TextEnhancer_Shutdown(session);
    dlclose(handle);

    std::cout << "Session shut down. Exiting." << std::endl;
    return 0;
}