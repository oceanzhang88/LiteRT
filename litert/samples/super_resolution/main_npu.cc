#include <iostream>
#include <vector>

#include "absl/types/span.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "image_utils.h"

// Export this function so it can be loaded from the .so
extern "C" int run_super_resolution_npu(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_path> <input_image_path> <output_image_path>"
                  << std::endl;
        return 1;
    }

  const std::string model_path = argv[1];
  const std::string input_file = argv[2];
  const std::string output_file = argv[3];

  // Initialize LiteRT environment and model
  LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));
  LITERT_ASSIGN_OR_ABORT(auto model, litert::Model::CreateFromFile(model_path));

  // Compile the model for the NPU
  LITERT_ASSIGN_OR_ABORT(litert::Options options, litert::Options::Create());
  options.SetHardwareAccelerators(kLiteRtHwAcceleratorNpu);
  LITERT_ASSIGN_OR_ABORT(auto compiled_model,
                         litert::CompiledModel::Create(env, model, options));

  // Create input and output buffers
  LITERT_ASSIGN_OR_ABORT(auto input_buffers, compiled_model.CreateInputBuffers());
  LITERT_ASSIGN_OR_ABORT(auto output_buffers, compiled_model.CreateOutputBuffers());

  // ================= PRE-PROCESSING =================
  // Load and preprocess the input image (CPU-based)
  int width_orig = 0, height_orig = 0, channels_file = 0, loaded_channels = 3;
  auto img_data_cpu = ImageUtils::LoadImage(input_file, width_orig, height_orig,
                                            channels_file, loaded_channels);
  if (!img_data_cpu) {
    std::cerr << "Failed to load image file: " << input_file << std::endl;
    return 1;
  }

  // Determine model input size
  LITERT_ASSIGN_OR_ABORT(auto input_tensor_type, model.GetInputTensorType(0, 0));
  int input_height = input_tensor_type.shape()[1];
  int input_width = input_tensor_type.shape()[2];

  // CPU-based resize
  std::vector<float> preprocessed_data(input_width * input_height * 3);
  for (int y = 0; y < input_height; ++y) {
    for (int x = 0; x < input_width; ++x) {
      for (int c = 0; c < 3; ++c) {
        preprocessed_data[(y * input_width + x) * 3 + c] =
            img_data_cpu[(y * width_orig + x) * 3 + c] / 255.0f;
      }
    }
  }
  ImageUtils::FreeImageData(img_data_cpu);
  LITERT_ABORT_IF_ERROR(
      input_buffers[0].Write(absl::MakeConstSpan(preprocessed_data)));

  // ================= INFERENCE =================
  LITERT_ABORT_IF_ERROR(
      compiled_model.Run(0, input_buffers, output_buffers));

  // ================= POST-PROCESSING =================
  // Determine model output size
  LITERT_ASSIGN_OR_ABORT(auto output_tensor_type, model.GetOutputTensorType(0, 0));
  int output_height = output_tensor_type.shape()[1];
  int output_width = output_tensor_type.shape()[2];
  int output_channels = output_tensor_type.shape()[3];

  // Read the output buffer
  std::vector<float> output_data(output_width * output_height *
                                 output_channels);
  LITERT_ABORT_IF_ERROR(output_buffers[0].Read(absl::MakeSpan(output_data)));

  // Convert float output to uchar for saving
  std::vector<unsigned char> output_uchar_data(output_data.size());
  for (size_t i = 0; i < output_data.size(); ++i) {
    output_uchar_data[i] = static_cast<unsigned char>(
        std::max(0.0f, std::min(1.0f, output_data[i])) * 255.0f);
  }

  // Save the output image
  if (!ImageUtils::SaveImage(output_file, output_width, output_height,
                             output_channels, output_uchar_data.data())) {
    std::cerr << "Failed to save the output image." << std::endl;
    return 1;
  }
  std::cout << "Successfully saved super-resolution image to " << output_file
            << std::endl;

  return 0;
}