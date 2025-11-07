#pragma once

#include <dlfcn.h>  // --- Include for dynamic loading ---

#include <algorithm>  // For std::min/max
#include <chrono>     // --- Include for timing ---
#include <iostream>
#include <string>
#include <vector>
#include <numeric>    // For std::accumulate
#include <iomanip>    // For std::setw, std::setprecision
#include <sstream>    // For std::ostringstream
#include <sys/stat.h> // For mkdir (Note: On Windows, you might need <direct.h> for _mkdir)

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
typedef TextEnhancerStatus (*t_TextEnhancer_PreProcess_AHB)(TextEnhancerSession* session, AHardwareBuffer* buffer);
#endif
typedef TextEnhancerStatus (*t_TextEnhancer_GetPreprocessedData)(TextEnhancerSession* session, uint8_t** data);
typedef TextEnhancerStatus (*t_TextEnhancer_PreProcess)(TextEnhancerSession* session, const uint8_t* rgb_data);
typedef TextEnhancerStatus (*t_TextEnhancer_Run)(TextEnhancerSession* session, float* inference_time_ms);
typedef TextEnhancerStatus (*t_TextEnhancer_PostProcess)(TextEnhancerSession* session, TextEnhancerOutput& output);
typedef void (*t_TextEnhancer_FreeOutputData)(TextEnhancerOutput& output);
// ----------------------------------------------------------------------

// --- RENAMED: Global function pointers ---
// (Static to avoid multiple definition errors when included in different .cc files)
static t_TextEnhancer_Initialize fn_TextEnhancer_Initialize = nullptr;
static t_TextEnhancer_Shutdown fn_TextEnhancer_Shutdown = nullptr;
#ifdef __ANDROID__
static t_TextEnhancer_PreProcess_AHB fn_TextEnhancer_PreProcess_AHB = nullptr;
#endif
static t_TextEnhancer_GetPreprocessedData fn_TextEnhancer_GetPreprocessedData = nullptr;
static t_TextEnhancer_PreProcess fn_TextEnhancer_PreProcess = nullptr;
static t_TextEnhancer_Run fn_TextEnhancer_Run = nullptr;
static t_TextEnhancer_PostProcess fn_TextEnhancer_PostProcess = nullptr;
static t_TextEnhancer_FreeOutputData fn_TextEnhancer_FreeOutputData = nullptr;
// -------------------------------------------------------------------

// Helper to parse command-line arguments (static to avoid multiple definitions)
static std::string GetFlagValue(int argc, char** argv, const std::string& flag, const std::string& default_value) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind(flag, 0) == 0) {
            return arg.substr(flag.length());
        }
    }
    return default_value;
}

// Common main function (inline to avoid multiple definitions)
inline int RunStandaloneSession(int argc, char** argv, const std::string& accelerator_name) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <lib_path.so> <model_path> <input_image> <output_image_base_path>"
                  << " [--preprocessor=cpu|vulkan]"
                  << " [--shader_path=path/to/shader]"
                  << " [--platform=desktop|android]"
                  << " [--save_preprocessed=true|false]" << std::endl;
        std::cerr << "Note: <output_image_base_path> will be used to generate output_run_images/basename_0.png, etc." << std::endl;
        return 1;
    }

    const char* lib_path = argv[1];
    const char* model_path = argv[2];
    const char* input_image_path = argv[3];
    const char* output_image_path = argv[4]; // This is now a base path

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
        std::cout << "Running on 'android' platform. Defaulting to Vulkan preprocessor." << std::endl;
    }
#else
    if (platform_str == "android") {
        std::cerr << "Error: --platform=android can only be used when compiled for Android." << std::endl;
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
        compute_shader_path_str = GetFlagValue(argc, argv, "--shader_path=", "shaders/crop_resize.spv");
        compute_shader_path = compute_shader_path_str.c_str();
        std::cout << "[Debug main] compute_shader_path set to: '" << compute_shader_path << "'" << std::endl;
    } else {
        std::cout << "[Debug main] compute_shader_path set to: '' (empty)" << std::endl;
    }

    // --- Load Image ---
    int img_width, img_height, img_channels;
    unsigned char* image_data_ptr = ImageUtils::LoadImage(input_image_path, img_width, img_height, img_channels, 4);
    if (!image_data_ptr) {
        std::cerr << "Failed to load image: " << input_image_path << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "Loaded input image: " << img_width << "x" << img_height << "x" << img_channels << std::endl;

    // --- Initialize the TextEnhancer session ---
    std::cout << "[Debug main] Calling TextEnhancer_Initialize..." << std::endl;

    // --- RENAMED: Struct types ---
    TextEnhancerOptions options = {};
    options.model_path = model_path;
    options.compute_shader_path = compute_shader_path;
    options.accelerator_name = accelerator_name.c_str();  // --- MODIFIED ---
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


    // --- Setup for 10-run test ---
    const int num_runs = 10;
    std::vector<double> all_preprocess_ms;
    std::vector<double> all_run_ms;
    std::vector<double> all_postprocess_ms;

    // --- Setup Output Directories ---
    std::string output_path_str = output_image_path;
    std::string output_filename = "output.png";
    size_t last_slash = output_path_str.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        output_filename = output_path_str.substr(last_slash + 1);
    } else {
        output_filename = output_path_str;
    }
    
    std::string output_run_dir = "output_run_images";
#ifdef _WIN32
    _mkdir(output_run_dir.c_str());
#else
    mkdir(output_run_dir.c_str(), 0755); // Create directory
#endif

    std::string output_base_name = "output";
    std::string output_extension = ".png";
    size_t last_dot = output_filename.find_last_of(".");
    if (last_dot != std::string::npos) {
        output_base_name = output_filename.substr(0, last_dot);
        output_extension = output_filename.substr(last_dot);
    }
    std::cout << "Saving " << num_runs << " output images to '" << output_run_dir << "' directory." << std::endl;


#ifdef __ANDROID__
    AHardwareBuffer* ahb_handle = nullptr;
    if (platform_str == "android") {
        std::cout << "Converting loaded image to AHardwareBuffer..." << std::endl;
        ahb_handle = ImageUtils::CreateAhbFromImageData(image_data_ptr, img_width, img_height);

        ImageUtils::FreeImageData(image_data_ptr);
        image_data_ptr = nullptr; // Set to null so we don't free it later

        if (!ahb_handle) {
            std::cerr << "Failed to create AHardwareBuffer from image data." << std::endl;
            fn_TextEnhancer_Shutdown(session);
            dlclose(handle);
            return 1;
        }
    }
#endif

    // --- Main Run Loop ---
    for (int i = 0; i < num_runs; ++i) {
        std::cout << "\n--- Run " << (i + 1) << "/" << num_runs << " ---" << std::endl;

        // --- Pre-process ---
        double preprocess_ms = 0.0;
        auto start_preprocess = std::chrono::high_resolution_clock::now();
        auto end_preprocess = std::chrono::high_resolution_clock::now();

#ifdef __ANDROID__
        if (platform_str == "android") {
            std::cout << "Pre-processing with AHardwareBuffer..." << std::endl;
            start_preprocess = std::chrono::high_resolution_clock::now();
            TextEnhancerStatus pre_status = fn_TextEnhancer_PreProcess_AHB(session, ahb_handle);
            end_preprocess = std::chrono::high_resolution_clock::now();
            preprocess_ms = std::chrono::duration<double, std::milli>(end_preprocess - start_preprocess).count();

            if (pre_status != kTextEnhancerOk) {
                std::cerr << "Pre-processing (AHB) failed." << std::endl;
                ImageUtils::FreeAhb(ahb_handle);
                fn_TextEnhancer_Shutdown(session);
                dlclose(handle);
                return 1;
            }
            std::cout << "Pre-processing (AHB) complete." << std::endl;

            if (save_preprocessed && i == 0) { // Only save first one
                std::cout << "Saving pre-processed image for verification..." << std::endl;
                uint8_t* pre_data = nullptr;
                TextEnhancerStatus status = fn_TextEnhancer_GetPreprocessedData(session, &pre_data);
                if (status == kTextEnhancerOk && pre_data) {
                    ImageUtils::SaveImage("preprocessed_output.png", options.input_width, options.input_height, 4, pre_data);
                    std::cout << "Pre-processed image saved to preprocessed_output.png" << std::endl;
                } else {
                    std::cerr << "Failed to get pre-processed data for saving." << std::endl;
                }
            }
        } else
#endif
        {
            // --- Desktop (CPU or Vulkan Staging) Path ---
            std::cout << "Pre-processing with CPU buffer..." << std::endl;
            start_preprocess = std::chrono::high_resolution_clock::now();
            TextEnhancerStatus pre_status = fn_TextEnhancer_PreProcess(session, image_data_ptr);
            end_preprocess = std::chrono::high_resolution_clock::now();
            preprocess_ms = std::chrono::duration<double, std::milli>(end_preprocess - start_preprocess).count();

            if (pre_status != kTextEnhancerOk) {
                std::cerr << "Pre-processing failed." << std::endl;
                // Don't free image_data_ptr here, will be freed after loop
                fn_TextEnhancer_Shutdown(session);
                dlclose(handle);
                return 1;
            }
            // DO NOT free image_data_ptr here, it's needed for the next loop iteration
            std::cout << "Pre-processing complete." << std::endl;

            if (save_preprocessed && i == 0) { // Only save first one
                std::cout << "Saving pre-processed image for verification..." << std::endl;
                uint8_t* pre_data = nullptr;
                TextEnhancerStatus status = fn_TextEnhancer_GetPreprocessedData(session, &pre_data);
                if (status == kTextEnhancerOk && pre_data) {
                    ImageUtils::SaveImage("preprocessed_output.png", options.input_width, options.input_height, 4, pre_data);
                    std::cout << "Pre-processed image saved to preprocessed_output.png" << std::endl;
                } else {
                    std::cerr << "Failed to get pre-processed data for saving." << std::endl;
                }
            }
        }
        all_preprocess_ms.push_back(preprocess_ms);

        // --- Run Inference ---
        float inference_time_ms = 0.0f;
        if (fn_TextEnhancer_Run(session, &inference_time_ms) != kTextEnhancerOk) {
            std::cerr << "Inference run failed." << std::endl;
#ifdef __ANDROID__
            if (ahb_handle) ImageUtils::FreeAhb(ahb_handle);
#endif
            if (image_data_ptr) ImageUtils::FreeImageData(image_data_ptr);
            fn_TextEnhancer_Shutdown(session);
            dlclose(handle);
            return 1;
        }
        all_run_ms.push_back(static_cast<double>(inference_time_ms));
        std::cout << "Inference complete." << std::endl;

        // --- Post-process & Save ---
        auto start_postprocess = std::chrono::high_resolution_clock::now();

        TextEnhancerOutput output_data = {0};
        if (fn_TextEnhancer_PostProcess(session, output_data) != kTextEnhancerOk) {
            std::cerr << "Post-processing failed." << std::endl;
#ifdef __ANDROID__
            if (ahb_handle) ImageUtils::FreeAhb(ahb_handle);
#endif
            if (image_data_ptr) ImageUtils::FreeImageData(image_data_ptr);
            fn_TextEnhancer_Shutdown(session);
            dlclose(handle);
            return 1;
        }

        auto end_postprocess = std::chrono::high_resolution_clock::now();
        double postprocess_ms = std::chrono::duration<double, std::milli>(end_postprocess - start_postprocess).count();
        all_postprocess_ms.push_back(postprocess_ms);
        std::cout << "Output received: " << output_data.width << "x" << output_data.height << std::endl;


        // --- E2E Timings for THIS run ---
        double run_ms = static_cast<double>(inference_time_ms);
        double total_e2e_ms = preprocess_ms + run_ms + postprocess_ms;

        std::cout << "--- Run " << (i + 1) << " Timing Summary ---" << std::endl;
        std::cout << "Pre-processing Time: " << preprocess_ms << " ms" << std::endl;
        std::cout << "Inference Time (TextEnhancer_Run): " << run_ms << " ms" << std::endl;
        std::cout << "Post-processing Time: " << postprocess_ms << " ms" << std::endl;
        std::cout << "Total E2E Time (Pre + Run + Post): " << total_e2e_ms << " ms" << std::endl;

        // --- Output Conversion Logic ---
        float* output_data_float = reinterpret_cast<float*>(output_data.data);
        std::vector<unsigned char> output_image_bytes(output_data.width * output_data.height * 3);
        for (int y = 0; y < output_data.height; ++y) {
            for (int x = 0; x < output_data.width; ++x) {
                int in_idx = (y * output_data.width + x) * output_data.channels;
                int out_idx = (y * output_data.width + x) * 3;

                output_image_bytes[out_idx + 0] =
                    static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, output_data_float[in_idx + 0] * 255.0f)));
                output_image_bytes[out_idx + 1] =
                    static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, output_data_float[in_idx + 1] * 255.0f)));
                output_image_bytes[out_idx + 2] =
                    static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, output_data_float[in_idx + 2] * 255.0f)));
            }
        }

        // --- Save This Run's Image ---
        std::ostringstream oss;
        oss << output_run_dir << "/" << output_base_name << "_" << i << output_extension;
        std::string current_output_path = oss.str();
        
        ImageUtils::SaveImage(current_output_path.c_str(), output_data.width, output_data.height, 3, output_image_bytes.data());
        std::cout << "Output image " << i << " saved to " << current_output_path << std::endl;

        // --- Free this run's output data ---
        fn_TextEnhancer_FreeOutputData(output_data);

    } // --- END of num_runs loop ---


    // --- Calculate and Print Final Statistics ---
    double sum_pre = std::accumulate(all_preprocess_ms.begin(), all_preprocess_ms.end(), 0.0);
    double sum_run = std::accumulate(all_run_ms.begin(), all_run_ms.end(), 0.0);
    double sum_post = std::accumulate(all_postprocess_ms.begin(), all_postprocess_ms.end(), 0.0);

    double avg_pre = sum_pre / num_runs;
    double avg_run = sum_run / num_runs;
    double avg_post = sum_post / num_runs;

    double min_pre = *std::min_element(all_preprocess_ms.begin(), all_preprocess_ms.end());
    double min_run = *std::min_element(all_run_ms.begin(), all_run_ms.end());
    double min_post = *std::min_element(all_postprocess_ms.begin(), all_postprocess_ms.end());

    double max_pre = *std::max_element(all_preprocess_ms.begin(), all_preprocess_ms.end());
    double max_run = *std::max_element(all_run_ms.begin(), all_run_ms.end());
    double max_post = *std::max_element(all_postprocess_ms.begin(), all_postprocess_ms.end());

    std::cout << "\n--- Timing Statistics (" << num_runs << " runs) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "                   Min (ms)   Max (ms)   Avg (ms)" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "Pre-processing:  " << std::setw(9) << min_pre
              << std::setw(11) << max_pre << std::setw(11) << avg_pre << std::endl;
    std::cout << "Inference (Run): " << std::setw(9) << min_run
              << std::setw(11) << max_run << std::setw(11) << avg_run << std::endl;
    std::cout << "Post-processing: " << std::setw(9) << min_post
              << std::setw(11) << max_post << std::setw(11) << avg_post << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "Total E2E (Avg): " << (avg_pre + avg_run + avg_post) << " ms" << std::endl;
    std::cout << "-------------------------------------------------\n" << std::endl;


    // --- Cleanup ---
#ifdef __ANDROID__
    if (ahb_handle) ImageUtils::FreeAhb(ahb_handle);
#endif
    if (image_data_ptr) ImageUtils::FreeImageData(image_data_ptr); // Now freed after loop

    // fn_TextEnhancer_FreeOutputData(output_data); // This is now done inside the loop
    fn_TextEnhancer_Shutdown(session);
    dlclose(handle);

    std::cout << "Session shut down. Exiting." << std::endl;
    return 0;
}