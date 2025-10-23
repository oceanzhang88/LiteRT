#include <algorithm>  // for std::max, std::min
#include <cctype>
#include <iostream>
#include <string>
#include <vector>

#include "image_utils.h"  // For Load/Save
#include "super_res_api.h"

int main(int argc, char* argv[]) {
  if (argc < 6 || argc > 7) {
    std::cerr << "Usage: " << argv[0]
              << " <model_path> <passthrough_shader.vert> "
                 "<super_res_compute.glsl> "
              << "<input_image_path> <output_image_path> [use_gl_buffers "
                 "(true|false)]"
              << std::endl;
    return 1;
  }

  const char* model_path = argv[1];
  const char* vert_shader_path = argv[2];
  const char* compute_shader_path = argv[3];
  const std::string input_file = argv[4];
  const std::string output_file = argv[5];

  bool use_gl_buffers = false;
  if (argc == 7) {
    std::string use_gl_buffers_arg = argv[6];
    std::transform(use_gl_buffers_arg.begin(), use_gl_buffers_arg.end(),
                   use_gl_buffers_arg.begin(), ::tolower);
    use_gl_buffers = (use_gl_buffers_arg == "true");
  }

  // 1. Initialize
  std::cout << "Initializing session..." << std::endl;
  SuperResSession* session = SuperRes_Initialize(
      model_path, vert_shader_path, compute_shader_path, use_gl_buffers);

  if (!session) {
    std::cerr << "Failed to initialize Super Resolution session." << std::endl;
    return 1;
  }
  std::cout << "Session initialized." << std::endl;

  // Load the input image
  ImageData input_image = {};
  int channels_file = 0;
  input_image.channels = 3;  // Request 3 channels (RGB)
  input_image.data = ImageUtils::LoadImage(
      input_file, input_image.width, input_image.height, channels_file,
      input_image.channels);

  if (!input_image.data) {
    std::cerr << "Failed to load image file: " << input_file << std::endl;
    SuperRes_Shutdown(session);
    return 1;
  }
  std::cout << "Input image loaded." << std::endl;

  // 2. Pre-process
  std::cout << "Pre-processing image..." << std::endl;
  if (!SuperRes_PreProcess(session, &input_image)) {
    std::cerr << "Failed to pre-process image." << std::endl;
    ImageUtils::FreeImageData(input_image.data);
    SuperRes_Shutdown(session);
    return 1;
  }
  std::cout << "Image pre-processed." << std::endl;

  // Free CPU-side image data now that it's on the GPU
  ImageUtils::FreeImageData(input_image.data);

  // 3. Run
  std::cout << "Running inference..." << std::endl;
  if (!SuperRes_Run(session)) {
    std::cerr << "Failed to run inference." << std::endl;
    SuperRes_Shutdown(session);
    return 1;
  }
  std::cout << "Inference complete." << std::endl;

  // 4. Post-process
  std::cout << "Post-processing result..." << std::endl;
  OutputData output_result = {};
  if (!SuperRes_PostProcess(session, &output_result)) {
    std::cerr << "Failed to post-process result." << std::endl;
    SuperRes_Shutdown(session);
    return 1;
  }
  std::cout << "Result post-processed." << std::endl;

  // Convert float output to uchar for saving
  size_t output_size =
      output_result.width * output_result.height * output_result.channels;
  std::vector<unsigned char> output_uchar_data(output_size);
  for (size_t i = 0; i < output_size; ++i) {
    output_uchar_data[i] = static_cast<unsigned char>(
        std::max(0.0f, std::min(1.0f, output_result.data[i])) * 255.0f);
  }

  // Save the output image
  if (!ImageUtils::SaveImage(output_file, output_result.width,
                            output_result.height, output_result.channels,
                            output_uchar_data.data())) {
    std::cerr << "Failed to save the output image." << std::endl;
    SuperRes_FreeOutputData(&output_result);
    SuperRes_Shutdown(session);
    return 1;
  }
  std::cout << "Successfully saved super-resolution image to " << output_file
            << std::endl;

  // 5. Shutdown
  std::cout << "Shutting down session..." << std::endl;
  SuperRes_FreeOutputData(&output_result);
  SuperRes_Shutdown(session);
  std::cout << "Session shut down." << std::endl;

  return 0;
}