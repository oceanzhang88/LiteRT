#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"

// Note: No ImageProcessor or GL headers needed
#include "super_res_api.h"
#include "image_utils.h" // For ImageUtils::ResizeImage


namespace {
// Creates LiteRT CPU options.
litert::Options CreateCpuOptions() {
  LITERT_ASSIGN_OR_ABORT(litert::Options options, litert::Options::Create());
  options.SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);
  return options;
}
}  // namespace

// The implementation of the opaque handle for CPU
struct SuperResSession {
  std::unique_ptr<litert::Environment> env;
  std::unique_ptr<litert::Model> model;
  std::unique_ptr<litert::CompiledModel> compiled_model;
  std::unique_ptr<std::vector<litert::TensorBuffer>> input_buffers;
  std::unique_ptr<std::vector<litert::TensorBuffer>> output_buffers;

  // Model dimensions
  int input_width = 0;
  int input_height = 0;
  int input_channels = 0;
  int output_width = 0;
  int output_height = 0;
  int output_channels = 0;
  
  // CPU buffer for pre-processed data
  std::vector<float> preprocessed_data;
};

extern "C" {

SuperResSession* SuperRes_Initialize(
    const char* model_path, const char* passthrough_vert_shader_path,
    const char* super_res_compute_shader_path, bool use_gl_buffers) {
  // Ignored parameters for CPU:
  // - passthrough_vert_shader_path
  // - super_res_compute_shader_path
  // - use_gl_buffers

  auto session = std::make_unique<SuperResSession>();

  LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));
  session->env = std::make_unique<litert::Environment>(std::move(env));

  LITERT_ASSIGN_OR_ABORT(auto model, litert::Model::CreateFromFile(model_path));

  // Get model dimensions
  LITERT_ASSIGN_OR_ABORT(auto input_tensor_type,
                         model.GetInputTensorType(0, 0));
  session->input_height = input_tensor_type.Layout().Dimensions()[1];
  session->input_width = input_tensor_type.Layout().Dimensions()[2];
  session->input_channels = input_tensor_type.Layout().Dimensions()[3];

  LITERT_ASSIGN_OR_ABORT(auto output_tensor_type,
                         model.GetOutputTensorType(0, 0));
  session->output_height = output_tensor_type.Layout().Dimensions()[1];
  session->output_width = output_tensor_type.Layout().Dimensions()[2];
  session->output_channels = output_tensor_type.Layout().Dimensions()[3];
  
  // Allocate the CPU buffer
  session->preprocessed_data.resize(session->input_width * session->input_height *
                                    session->input_channels);

  session->model = std::make_unique<litert::Model>(std::move(model));

  litert::Options options = CreateCpuOptions();
  LITERT_ASSIGN_OR_ABORT(
      auto compiled_model,
      litert::CompiledModel::Create(*session->env, *session->model, options));
  session->compiled_model =
      std::make_unique<litert::CompiledModel>(std::move(compiled_model));

  LITERT_ASSIGN_OR_ABORT(auto input_buffers,
                         session->compiled_model->CreateInputBuffers());
  session->input_buffers =
      std::make_unique<std::vector<litert::TensorBuffer>>(std::move(input_buffers));

  LITERT_ASSIGN_OR_ABORT(auto output_buffers,
                         session->compiled_model->CreateOutputBuffers());
  session->output_buffers =
      std::make_unique<std::vector<litert::TensorBuffer>>(std::move(output_buffers));

  return session.release();  // Transfer ownership to the caller
}

void SuperRes_Shutdown(SuperResSession* session) {
  if (!session) return;
  // Smart pointers handle all cleanup
  delete session;
}

bool SuperRes_PreProcess(SuperResSession* session,
                         const ImageData* input_image) {
  if (!session || !input_image || !input_image->data) return false;

  const unsigned char* image_data_ptr = input_image->data;
  bool needs_free = false;

  // Check if resizing is needed
  if (input_image->width != session->input_width ||
      input_image->height != session->input_height) {
    std::cout << "Resizing input image from " << input_image->width << "x"
              << input_image->height << " to " << session->input_width << "x"
              << session->input_height << std::endl;
              
    image_data_ptr = ImageUtils::ResizeImage(
        input_image->data, input_image->width, input_image->height,
        input_image->channels, session->input_width, session->input_height);
        
    if (!image_data_ptr) {
      std::cerr << "Failed to resize input image." << std::endl;
      return false;
    }
    needs_free = true; // We allocated this buffer and must free it
  }
  
  // CPU-side pre-processing: uint8 to float normalization
  size_t image_size =
      session->input_width * session->input_height * session->input_channels;
  for (size_t i = 0; i < image_size; ++i) {
    session->preprocessed_data[i] =
        static_cast<float>(image_data_ptr[i]) / 255.0f;
  }
  
  // Free the intermediate resized buffer if we created one
  if (needs_free) {
    // Note: ImageUtils::FreeImageData uses stbi_image_free, 
    // but we used new[]. We should use delete[].
    delete[] image_data_ptr; 
    // To be more robust, ImageUtils::FreeImageData should be
    // adjusted or we should have a separate free function.
    // For now, delete[] is correct for the buffer from ResizeImage.
  }

  // Write to the LiteRT input buffer
  LITERT_ABORT_IF_ERROR((*session->input_buffers)[0].Write(
      absl::MakeConstSpan(session->preprocessed_data)));

  return true;
}

bool SuperRes_Run(SuperResSession* session) {
  if (!session) return false;

  bool async = false;
  LITERT_ABORT_IF_ERROR(session->compiled_model->RunAsync(
      0, *session->input_buffers, *session->output_buffers, async));

  return true;
}

bool SuperRes_PostProcess(SuperResSession* session, OutputData* output_data) {
  if (!session || !output_data) return false;

  if ((*session->output_buffers)[0].HasEvent()) {
    LITERT_ASSIGN_OR_ABORT(auto event, (*session->output_buffers)[0].GetEvent());
    event.Wait();
  }

  size_t output_size =
      session->output_width * session->output_height * session->output_channels;

  // We use a vector first to read the data
  std::vector<float> output_vec(output_size);
  LITERT_ABORT_IF_ERROR(
      (*session->output_buffers)[0].Read(absl::MakeSpan(output_vec)));

  // Allocate memory for the output data buffer for the C-API
  float* data_ptr = new (std::nothrow) float[output_size];
  if (!data_ptr) {
    std::cerr << "Failed to allocate memory for output data." << std::endl;
    return false;
  }

  // Copy data to the allocated buffer
  memcpy(data_ptr, output_vec.data(), output_size * sizeof(float));

  output_data->data = data_ptr;
  output_data->width = session->output_width;
  output_data->height = session->output_height;
  output_data->channels = session->output_channels;

  return true;
}

void SuperRes_FreeOutputData(OutputData* output_data) {
  if (output_data && output_data->data) {
    delete[] output_data->data;
    output_data->data = nullptr;
  }
}

}  // extern "C"