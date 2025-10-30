#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm> // For std::min/max

#include "super_res_api.h"
#include "utils/image_utils.h"

// Include AHB headers if compiling for Android
#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#endif

// Helper to parse command-line arguments
std::string GetFlagValue(int argc, char** argv, const std::string& flag, const std::string& default_value) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind(flag, 0) == 0) {
            return arg.substr(flag.length());
        }
    }
    return default_value;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_path> <input_image> <output_image>"
                  << " [--preprocessor=cpu|vulkan]"
                  << " [--shader_path=path/to/shader]"
                  << " [--platform=desktop|android]"
                  << " [--save_preprocessed=true|false]" // <-- NEW FLAG
                  << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    const char* input_image_path = argv[2];
    const char* output_image_path = argv[3];

    // --- Flag Parsing ---
    std::string platform_str = GetFlagValue(argc, argv, "--platform=", "android");
    std::string preprocessor_type_str = GetFlagValue(argc, argv, "--preprocessor=", "vulkan");
    // --- NEW: Parse save_preprocessed flag ---
    std::string save_preprocessed_str = GetFlagValue(argc, argv, "--save_preprocessed=", "true");
    bool save_preprocessed = (save_preprocessed_str == "true");
    // -----------------------------------------
    SuperRes_PreprocessorType preprocessor_type = kSuperResCpuPreprocessor;
    
    std::string compute_shader_path_str = ""; 
    const char* compute_shader_path = ""; 

    // --- Platform-Specific Logic ---
#ifdef __ANDROID__
    if (platform_str == "android") {
        // On Android platform, force Vulkan preprocessor
        preprocessor_type = kSuperResVulkanPreprocessor;
        std::cout << "Running on 'android' platform. Forcing Vulkan preprocessor." << std::endl;
    }
#else
    if (platform_str == "android") {
        std::cerr << "Error: --platform=android can only be used when compiled for Android." << std::endl;
        return 1;
    }
#endif

    // --- Preprocessor Logic (for desktop platform) ---
    if (platform_str == "desktop") {
        if (preprocessor_type_str == "vulkan") {
            preprocessor_type = kSuperResVulkanPreprocessor;
            std::cout << "Using Vulkan Pre-processor (Staging Buffer Path)" << std::endl;
        } else {
            preprocessor_type = kSuperResCpuPreprocessor;
            std::cout << "Using CPU Pre-processor" << std::endl;
        }
    }

    // --- Get Shader Path (if Vulkan is used) ---
    if (preprocessor_type == kSuperResVulkanPreprocessor) {
        compute_shader_path_str = GetFlagValue(argc, argv, "--shader_path=", "shaders/crop_resize.spv");
        compute_shader_path = compute_shader_path_str.c_str();
        std::cout << "[Debug main] compute_shader_path set to: '" << compute_shader_path << "'" << std::endl;
    } else {
        std::cout << "[Debug main] compute_shader_path set to: '' (empty)" << std::endl;
    }
    // -------------------------

    // --- Initialize the SuperRes session ---
    std::cout << "[Debug main] Calling SuperRes_Initialize with shader_path: '" 
              << (compute_shader_path ? compute_shader_path : "NULL") << "'" << std::endl;

    SuperResSession* session = SuperRes_Initialize(
        model_path,
        preprocessor_type,
        "", // passthrough_vert_shader_path
        compute_shader_path
    );
    
    if (!session) {
        std::cerr << "Failed to initialize SuperRes session." << std::endl;
        return 1;
    }
    std::cout << "SuperRes session initialized." << std::endl;

    // --- Load Image ---
    int img_width, img_height, img_channels;
    // Force 4 channels (RGBA) as that's what our pre-processors expect
    unsigned char* image_data_ptr = ImageUtils::LoadImage(
        input_image_path, img_width, img_height, img_channels, 4); 
    if (!image_data_ptr) {
        std::cerr << "Failed to load image: " << input_image_path << std::endl;
        SuperRes_Shutdown(session);
        return 1;
    }
    std::cout << "Loaded input image: " << img_width << "x" << img_height << "x" << img_channels << std::endl;
    
    // --- Pre-process ---
    // This is where the new platform logic splits
    
#ifdef __ANDROID__
    AHardwareBuffer* ahb_handle = nullptr;
    if (platform_str == "android") {
        // --- Android AHB Path ---
        std::cout << "Converting loaded image to AHardwareBuffer..." << std::endl;
        ahb_handle = ImageUtils::CreateAhbFromImageData(image_data_ptr, img_width, img_height);
        
        // We are done with the CPU-side image data
        ImageUtils::FreeImageData(image_data_ptr);
        image_data_ptr = nullptr; // Mark as freed

        if (!ahb_handle) {
            std::cerr << "Failed to create AHardwareBuffer from image data." << std::endl;
            SuperRes_Shutdown(session);
            return 1;
        }

        std::cout << "Pre-processing with AHardwareBuffer..." << std::endl;
        
        // --- FIX: Renamed function call ---
        if (!SuperRes_PreProcess_AHB(session, ahb_handle, img_width, img_height)) { // <-- RENAMED
            std::cerr << "Pre-processing (AHB) failed." << std::endl;
            ImageUtils::FreeAhb(ahb_handle);
            SuperRes_Shutdown(session);
            return 1;
        }
        std::cout << "Pre-processing (AHB) complete." << std::endl;

        // --- NEW: Save pre-processed image if flag is set (Android path) ---
        if (save_preprocessed) {
            std::cout << "Saving pre-processed image for verification..." << std::endl;
            int pre_width, pre_height, pre_channels;
            const float* pre_data = SuperRes_GetPreprocessedData(session, &pre_width, &pre_height, &pre_channels);

            if (pre_data) {
                std::vector<unsigned char> pre_image_bytes(pre_width * pre_height * pre_channels);
                for (int i = 0; i < pre_width * pre_height * pre_channels; ++i) {
                    pre_image_bytes[i] = static_cast<unsigned char>(
                        std::max(0.0f, std::min(255.0f, pre_data[i] * 255.0f))
                    );
                }
                std::string pre_output_path = "preprocessed_output.png"; // Note: on-device path
                ImageUtils::SaveImage(pre_output_path, pre_width, pre_height,
                                      pre_channels, pre_image_bytes.data());
                std::cout << "Pre-processed image saved to " << pre_output_path << std::endl;
            } else {
                std::cerr << "Failed to get pre-processed data for saving." << std::endl;
            }
        }
        // --------------------------------------------------

    } else
#endif
    {
        // --- Desktop (CPU or Vulkan Staging) Path ---
        ImageData input_image = {image_data_ptr, img_width, img_height, 4};
        
        std::cout << "Pre-processing with CPU buffer..." << std::endl;
        if (!SuperRes_PreProcess(session, &input_image)) {
            std::cerr << "Pre-processing failed." << std::endl;
            ImageUtils::FreeImageData(image_data_ptr);
            SuperRes_Shutdown(session);
            return 1;
        }
        ImageUtils::FreeImageData(image_data_ptr); // Free the original image data
        image_data_ptr = nullptr; // Mark as freed
        std::cout << "Pre-processing complete." << std::endl;

        // --- NEW: Save pre-processed image if flag is set (Desktop path) ---
        if (save_preprocessed) {
            std::cout << "Saving pre-processed image for verification..." << std::endl;
            int pre_width, pre_height, pre_channels;
            const float* pre_data = SuperRes_GetPreprocessedData(session, &pre_width, &pre_height, &pre_channels);

            if (pre_data) {
                // Convert float buffer back to uchar for saving
                std::vector<unsigned char> pre_image_bytes(pre_width * pre_height * pre_channels);
                for (int i = 0; i < pre_width * pre_height * pre_channels; ++i) {
                    // De-normalize [0.0, 1.0] float to [0, 255] uchar
                    pre_image_bytes[i] = static_cast<unsigned char>(
                        std::max(0.0f, std::min(255.0f, pre_data[i] * 255.0f))
                    );
                }

                // Determine output path
                std::string pre_output_path = "preprocessed_output.png";
                ImageUtils::SaveImage(pre_output_path, pre_width, pre_height,
                                      pre_channels, pre_image_bytes.data());
                std::cout << "Pre-processed image saved to " << pre_output_path << std::endl;
            } else {
                std::cerr << "Failed to get pre-processed data for saving." << std::endl;
            }
        }
        // --------------------------------------------------
    }

    // --- Run Inference ---
    if (!SuperRes_Run(session)) {
        std::cerr << "Inference run failed." << std::endl;
#ifdef __ANDROID__
        if (ahb_handle) ImageUtils::FreeAhb(ahb_handle);
#endif
        SuperRes_Shutdown(session);
        return 1;
    }
    std::cout << "Inference complete." << std::endl;

    // --- Post-process & Save ---
    OutputData output_data = {0};
    if (!SuperRes_PostProcess(session, &output_data)) {
        std::cerr << "Post-processing failed." << std::endl;
#ifdef __ANDROID__
        if (ahb_handle) ImageUtils::FreeAhb(ahb_handle);
#endif
        SuperRes_Shutdown(session);
        return 1;
    }

    std::cout << "Output received: " << output_data.width << "x" << output_data.height << std::endl;

    // Convert float output (e.g., RGBA) to unsigned char (RGB) for saving
    std::vector<unsigned char> output_image_bytes(output_data.width * output_data.height * 3); // Save as 3-channel RGB
    for (int y = 0; y < output_data.height; ++y) {
        for (int x = 0; x < output_data.width; ++x) {
            int in_idx = (y * output_data.width + x) * output_data.channels;
            int out_idx = (y * output_data.width + x) * 3;
            
            // Clamp and convert [0.0, 1.0] float to [0, 255] uchar
            // Save as RGB, discarding alpha if present
            output_image_bytes[out_idx + 0] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, output_data.data[in_idx + 0] * 255.0f)));
            output_image_bytes[out_idx + 1] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, output_data.data[in_idx + 1] * 255.0f)));
            output_image_bytes[out_idx + 2] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, output_data.data[in_idx + 2] * 255.0f)));
        }
    }

    ImageUtils::SaveImage(output_image_path, output_data.width, output_data.height,
                          3, output_image_bytes.data()); // Save 3 channels

    std::cout << "Output image saved to " << output_image_path << std::endl;

    // --- Cleanup ---
#ifdef __ANDROID__
    if (ahb_handle) ImageUtils::FreeAhb(ahb_handle);
#endif
    if (image_data_ptr) ImageUtils::FreeImageData(image_data_ptr); // Just in case of error path
    SuperRes_FreeOutputData(&output_data);
    SuperRes_Shutdown(session);
    std::cout << "Session shut down. Exiting." << std::endl;
    return 0;
}