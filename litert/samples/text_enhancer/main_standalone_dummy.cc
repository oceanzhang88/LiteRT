#include <iostream>
#include <string>
#include <vector>

#include "super_res_api.h"
#include "utils/image_utils.h"

int main(int argc, char** argv) {
    std::cout << "--- Running Super-Res Standalone (Dummy) ---" << std::endl;

    const char* output_path = "dummy_checkerboard_output.png";

    // 1. Initialize the session.
    // Paths are ignored by the dummy implementation.
    std::cout << "Initializing dummy session..." << std::endl;
    SuperResSession* session = SuperRes_Initialize(
        "", kSuperResCpuPreprocessor, "", "");

    if (!session) {
        std::cerr << "Failed to initialize SuperResSession." << std::endl;
        return -1;
    }

    // 2. Pre-processing (optional, but good to test the call)
    // We pass a null ImageData pointer since the dummy backend ignores it.
    std::cout << "Calling dummy pre-process..." << std::endl;
    if (!SuperRes_PreProcess(session, nullptr)) {
        std::cerr << "Dummy pre-process call failed." << std::endl;
        SuperRes_Shutdown(session);
        return -1;
    }

    // 3. Run inference
    std::cout << "Calling dummy run..." << std::endl;
    if (!SuperRes_Run(session)) {
        std::cerr << "Dummy run call failed." << std::endl;
        SuperRes_Shutdown(session);
        return -1;
    }

    // 4. Post-process to get the checkerboard
    std::cout << "Calling dummy post-process..." << std::endl;
    OutputData output_data = {};
    if (!SuperRes_PostProcess(session, &output_data)) {
        std::cerr << "Dummy post-process call failed." << std::endl;
        SuperRes_Shutdown(session);
        return -1;
    }

    std::cout << "Got output data: "
              << output_data.width << "x"
              << output_data.height << "x"
              << output_data.channels << std::endl;

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

    // 5. Save the output
    std::cout << "Saving output to " << output_path << "..." << std::endl;
    if (!ImageUtils::SaveImage(output_path, output_data.width, output_data.height,
                          3, output_image_bytes.data())) {
        std::cerr << "Failed to save output image." << std::endl;
    } else {
        std::cout << "Successfully saved dummy output!" << std::endl;
    }

    // 6. Clean up
    SuperRes_FreeOutputData(&output_data);
    SuperRes_Shutdown(session);

    std::cout << "--------------------------------------------" << std::endl;
    return 0;
}