#include <iostream>
#include <memory>  // For std::unique_ptr
#include <string>

#include "absl/types/span.h"
#include "image_processor.h"  // Assumed to be in include path
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "super_res_api.h"

namespace {
// Helper function to create LiteRT GPU options.
litert::Options CreateGpuOptions(bool use_gl_buffers) {
    LITERT_ASSIGN_OR_ABORT(auto gpu_options, litert::GpuOptions::Create());
     LITERT_ABORT_IF_ERROR(gpu_options.SetGpuBackend(kLiteRtGpuBackendOpenCl)); //Android kLiteRtGpuBackendOpenCl

    if (use_gl_buffers) {
        LITERT_ABORT_IF_ERROR(gpu_options.SetDelegatePrecision(kLiteRtDelegatePrecisionFp32));
        LITERT_ABORT_IF_ERROR(gpu_options.SetBufferStorageType(kLiteRtDelegateBufferStorageTypeBuffer));
        LITERT_ABORT_IF_ERROR(gpu_options.EnableExternalTensorsMode(true));
    } else {
        LITERT_ABORT_IF_ERROR(gpu_options.EnableExternalTensorsMode(false));
    }
    LITERT_ASSIGN_OR_ABORT(litert::Options options, litert::Options::Create());
    options.SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
    options.AddOpaqueOptions(std::move(gpu_options));
    return options;
}
}  // namespace

// The implementation of the opaque handle
struct SuperResSession {
    std::unique_ptr<ImageProcessor> processor;
    std::unique_ptr<litert::Environment> env;
    std::unique_ptr<litert::Model> model;
    std::unique_ptr<litert::CompiledModel> compiled_model;
    std::unique_ptr<std::vector<litert::TensorBuffer>> input_buffers;
    std::unique_ptr<std::vector<litert::TensorBuffer>> output_buffers;

    // OpenGL resources
    GLuint tex_id_orig = 0;
    GLuint preprocessed_buffer_id = 0;

    // Model dimensions
    int input_width = 0;
    int input_height = 0;
    int input_channels = 0;
    int output_width = 0;
    int output_height = 0;
    int output_channels = 0;
};

extern "C" {

SuperResSession* SuperRes_Initialize(const char* model_path,
                                     const char* passthrough_vert_shader_path,
                                     const char* super_res_compute_shader_path,
                                     bool use_gl_buffers) {
    auto session = std::make_unique<SuperResSession>();

    session->processor = std::make_unique<ImageProcessor>();
    if (!session->processor->InitializeGL(passthrough_vert_shader_path,
                                          super_res_compute_shader_path)) {
        std::cerr << "Failed to initialize ImageProcessor." << std::endl;
        return nullptr;
    }

    LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));
    session->env = std::make_unique<litert::Environment>(std::move(env));

    LITERT_ASSIGN_OR_ABORT(auto model, litert::Model::CreateFromFile(model_path));

    // Get model dimensions
    LITERT_ASSIGN_OR_ABORT(auto input_tensor_type, model.GetInputTensorType(0, 0));
    session->input_height = input_tensor_type.Layout().Dimensions()[1];
    session->input_width = input_tensor_type.Layout().Dimensions()[2];
    session->input_channels = input_tensor_type.Layout().Dimensions()[3];

    LITERT_ASSIGN_OR_ABORT(auto output_tensor_type, model.GetOutputTensorType(0, 0));
    session->output_height = output_tensor_type.Layout().Dimensions()[1];
    session->output_width = output_tensor_type.Layout().Dimensions()[2];
    session->output_channels = output_tensor_type.Layout().Dimensions()[3];

    session->model = std::make_unique<litert::Model>(std::move(model));

    litert::Options options = CreateGpuOptions(use_gl_buffers);
    LITERT_ASSIGN_OR_ABORT(auto compiled_model,
                           litert::CompiledModel::Create(*session->env, *session->model, options));
    session->compiled_model = std::make_unique<litert::CompiledModel>(std::move(compiled_model));

    LITERT_ASSIGN_OR_ABORT(auto input_buffers, session->compiled_model->CreateInputBuffers());
    session->input_buffers =
        std::make_unique<std::vector<litert::TensorBuffer>>(std::move(input_buffers));

    LITERT_ASSIGN_OR_ABORT(auto output_buffers, session->compiled_model->CreateOutputBuffers());
    session->output_buffers =
        std::make_unique<std::vector<litert::TensorBuffer>>(std::move(output_buffers));

    // Create the GL buffer for preprocessing
    size_t buffer_size =
        session->input_width * session->input_height * session->input_channels * sizeof(float);
    session->preprocessed_buffer_id = session->processor->CreateOpenGLBuffer(nullptr, buffer_size);
    if (session->preprocessed_buffer_id == 0) {
        std::cerr << "Failed to create OpenGL preprocessed buffer." << std::endl;
        return nullptr;  // Initialization failed
    }

    return session.release();  // Transfer ownership to the caller
}

void SuperRes_Shutdown(SuperResSession* session) {
    if (!session) return;

    // Cleanup GL resources
    if (session->processor) {
        if (session->tex_id_orig != 0) {
            session->processor->DeleteOpenGLTexture(session->tex_id_orig);
        }
        if (session->preprocessed_buffer_id != 0) {
            session->processor->DeleteOpenGLBuffer(session->preprocessed_buffer_id);
        }
        // ImageProcessor's destructor calls ShutdownGL()
    }

    // Let smart pointers handle the rest of the cleanup
    delete session;
}

bool SuperRes_PreProcess(SuperResSession* session, const ImageData* input_image) {
    if (!session || !input_image || !input_image->data) return false;

    // Clean up previous texture if it exists
    if (session->tex_id_orig != 0) {
        session->processor->DeleteOpenGLTexture(session->tex_id_orig);
        session->tex_id_orig = 0;
    }

    session->tex_id_orig = session->processor->CreateOpenGLTexture(
        input_image->data, input_image->width, input_image->height, input_image->channels);

    if (!session->tex_id_orig) {
        std::cerr << "Failed to create OpenGL texture for image" << std::endl;
        return false;
    }

    // Check if resizing is needed
    if (input_image->width != session->input_width ||
        input_image->height != session->input_height) {
        std::cout << "Resizing input image from " << input_image->width << "x"
                  << input_image->height << " to " << session->input_width << "x"
                  << session->input_height << std::endl;
    }

    if (!session->processor->PreprocessInputForSuperResolution(
            session->tex_id_orig, session->input_width, session->input_height,
            session->preprocessed_buffer_id)) {
        std::cerr << "Failed to preprocess input image." << std::endl;
        return false;
    }

    // Read data from GL buffer into a CPU-side vector
    // This is needed for the input_buffers.Write call
    std::vector<float> preprocessed_data(session->input_width * session->input_height *
                                         session->input_channels);
    if (!session->processor->ReadBufferData(session->preprocessed_buffer_id, 0,
                                            preprocessed_data.size() * sizeof(float),
                                            preprocessed_data.data())) {
        std::cerr << "Failed to read preprocessed data from buffer." << std::endl;
        return false;
    }

    // Write to the LiteRT input buffer
    LITERT_ABORT_IF_ERROR(
        (*session->input_buffers)[0].Write(absl::MakeConstSpan(preprocessed_data)));

    return true;
}

bool SuperRes_Run(SuperResSession* session) {
    if (!session) return false;

    bool async = false;
    LITERT_ABORT_IF_ERROR(
        session->compiled_model->Run(*session->input_buffers, *session->output_buffers));

    return true;
}

bool SuperRes_PostProcess(SuperResSession* session, OutputData* output_data) {
    if (!session || !output_data) return false;

    if ((*session->output_buffers)[0].HasEvent()) {
        LITERT_ASSIGN_OR_ABORT(auto event, (*session->output_buffers)[0].GetEvent());
        event.Wait();
    }

    size_t output_size = session->output_width * session->output_height * session->output_channels;

    // We use a vector first to read the data
    std::vector<float> output_vec(output_size);
    LITERT_ABORT_IF_ERROR((*session->output_buffers)[0].Read(absl::MakeSpan(output_vec)));

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