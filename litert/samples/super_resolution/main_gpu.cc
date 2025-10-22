#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "image_processor.h"
#include "image_utils.h"

namespace {

// Creates LiteRT GPU options.
    litert::Options CreateGpuOptions(bool use_gl_buffers) {
        LITERT_ASSIGN_OR_ABORT(auto gpu_options, litert::GpuOptions::Create());
        if (use_gl_buffers) {
            LITERT_ABORT_IF_ERROR(gpu_options.EnableExternalTensorsMode(true));
        }
        LITERT_ASSIGN_OR_ABORT(litert::Options options, litert::Options::Create());
        options.SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
        options.AddOpaqueOptions(std::move(gpu_options));
        return options;
    }

}  // namespace

// Export this function so it can be loaded from the .so
extern "C" int run_super_resolution(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_path> <input_image_path> <output_image_path> "
                     "[use_gl_buffers (true|false)]"
                  << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string input_file = argv[2];
    const std::string output_file = argv[3];
    bool use_gl_buffers = false;
    if (argc == 5) {
        std::string use_gl_buffers_arg = argv[4];
        std::transform(use_gl_buffers_arg.begin(), use_gl_buffers_arg.end(),
                       use_gl_buffers_arg.begin(), ::tolower);
        use_gl_buffers = (use_gl_buffers_arg == "true");
    }

    // Initialize the ImageProcessor
    ImageProcessor processor;
    if (!processor.InitializeGL("shaders/passthrough_shader.vert",
                                "shaders/super_res_compute.glsl")) {
        std::cerr << "Failed to initialize ImageProcessor." << std::endl;
        return 1;
    }

    // Initialize LiteRT environment and model
    LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));
    LITERT_ASSIGN_OR_ABORT(auto model, litert::Model::CreateFromFile(model_path));

    // Compile the model for the GPU
    litert::Options options = CreateGpuOptions(use_gl_buffers);
    LITERT_ASSIGN_OR_ABORT(auto compiled_model,
                           litert::CompiledModel::Create(env, model, options));

    // Create input and output buffers
    LITERT_ASSIGN_OR_ABORT(auto input_buffers, compiled_model.CreateInputBuffers());
    LITERT_ASSIGN_OR_ABORT(auto output_buffers, compiled_model.CreateOutputBuffers());

    // ================= PRE-PROCESSING =================
    // Load the input image
    int width_orig = 0, height_orig = 0, channels_file = 0, loaded_channels = 3;
    auto img_data_cpu = ImageUtils::LoadImage(input_file, width_orig, height_orig,
                                              channels_file, loaded_channels);
    if (!img_data_cpu) {
        std::cerr << "Failed to load image file: " << input_file << std::endl;
        return 1;
    }
    GLuint tex_id_orig = processor.CreateOpenGLTexture(img_data_cpu, width_orig,
                                                       height_orig, loaded_channels);
    ImageUtils::FreeImageData(img_data_cpu);
    if (!tex_id_orig) {
        std::cerr << "Failed to create OpenGL texture for image" << std::endl;
        return 1;
    }

    // Determine model input size
    LITERT_ASSIGN_OR_ABORT(auto input_tensor_type, model.GetInputTensorType(0, 0));
    int input_height = input_tensor_type.shape()[1];
    int input_width = input_tensor_type.shape()[2];
    int input_channels = 3;

    // Create and preprocess the input buffer
    GLuint preprocessed_buffer_id = processor.CreateOpenGLBuffer(
            nullptr, input_width * input_height * input_channels * sizeof(float));

    // Updated function call
    if (!processor.PreprocessInputForSuperResolution(
            tex_id_orig, input_width, input_height,
            preprocessed_buffer_id)) {
        std::cerr << "Failed to preprocess input image." << std::endl;
        return 1;
    }

    std::vector<float> preprocessed_data(input_width * input_height *
                                         input_channels);
    processor.ReadBufferData(preprocessed_buffer_id, 0,
                             preprocessed_data.size() * sizeof(float),
                             preprocessed_data.data());
    LITERT_ABORT_IF_ERROR(
            input_buffers[0].Write(absl::MakeConstSpan(preprocessed_data)));

    // ================= INFERENCE =================
    bool async = false;
    LITERT_ABORT_IF_ERROR(
            compiled_model.RunAsync(0, input_buffers, output_buffers, async));

    // ================= POST-PROCESSING =================
    if (output_buffers[0].HasEvent()) {
        LITERT_ASSIGN_OR_ABORT(auto event, output_buffers[0].GetEvent());
        event.Wait();
    }

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

    // Cleanup
    processor.DeleteOpenGLTexture(tex_id_orig);
    processor.DeleteOpenGLBuffer(preprocessed_buffer_id);

    return 0;
}