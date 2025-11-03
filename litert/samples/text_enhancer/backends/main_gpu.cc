#include <iostream>
#include <memory>  // For std::unique_ptr
#include <string>
#include <vector> // --- NEW ---

#include "absl/types/span.h"
// #include "image_processor.h"  // --- REMOVED ---
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"

// --- NEW: Include headers from main_cpu.cc ---
#include "litert/cc/litert_profiler.h"
#include "litert/cc/options/litert_runtime_options.h"
// ---------------------------------------------

// --- NEW: Include AHB header for AHB API implementation ---
#ifdef __ANDROID__
#include "android/hardware_buffer.h"
#endif
// ---------------------------------------------------------
#include "litert/samples/super_resolution/super_res_api.h"
#include "litert/samples/super_resolution/utils/image_utils.h" // For ImageUtils::ResizeImageBilinear
#include "litert/samples/super_resolution/image_processing/vulkan_image_processor.h"


namespace {
// --- MODIFIED: CreateGpuOptions now hardcodes use_gl_buffers ---
// Helper function to create LiteRT GPU options.
litert::Options CreateGpuOptions() {
    // Hardcode use_gl_buffers = false, as this is no longer passed by the API
    bool use_gl_buffers = false;

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

// --- MODIFIED: SuperResSession struct to match main_cpu.cc ---
// The implementation of the opaque handle
struct SuperResSession {
    // std::unique_ptr<ImageProcessor> processor; // --- REMOVED ---
    std::unique_ptr<litert::Environment> env;
    std::unique_ptr<litert::Model> model;
    std::unique_ptr<litert::CompiledModel> compiled_model;
    std::unique_ptr<std::vector<litert::TensorBuffer>> input_buffers;
    std::unique_ptr<std::vector<litert::TensorBuffer>> output_buffers;

    // --- REMOVED: OpenGL resources ---
    // GLuint tex_id_orig = 0;
    // GLuint preprocessed_buffer_id = 0;

    // Model dimensions
    int input_width = 0;
    int input_height = 0;
    int input_channels = 0;
    int output_width = 0;
    int output_height = 0;
    int output_channels = 0;

    // --- NEW: Pre-processing fields from main_cpu.cc ---
    SuperRes_PreprocessorType preprocessor_type;
    void* processor;
    std::vector<float> preprocessed_data;
    std::vector<float> vulkan_temp_buffer;
    // ----------------------------------------------------
};

extern "C" {

// --- MODIFIED: SuperRes_Initialize signature and body ---
SuperResSession* SuperRes_Initialize(
    const char* model_path,
    SuperRes_PreprocessorType preprocessor_type,
    const char* passthrough_vert_shader_path,
    const char* compute_shader_path) {

    auto session = std::make_unique<SuperResSession>();
    session->preprocessor_type = preprocessor_type;
    session->processor = nullptr;

    // --- REMOVED: OpenGL ImageProcessor init ---
    // session->processor = std::make_unique<ImageProcessor>();
    // ...

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

    std::cout << "[Debug SuperRes_Initialize_GPU] Model Input: " 
              << session->input_width << "x" << session->input_height << "x" << session->input_channels << std::endl;

    // --- NEW: Copied pre-processor logic from main_cpu.cc ---
    session->preprocessed_data.resize(session->input_width * session->input_height *
                                      session->input_channels);

    if (session->preprocessor_type == kSuperResVulkanPreprocessor) {
        std::cout << "Initializing Vulkan Pre-processor..." << std::endl;
        auto vk_processor = std::make_unique<VulkanImageProcessor>();
        
        if (session->input_width != 256 || session->input_height != 256) {
             std::cerr << "Warning: Vulkan pre-processor is hard-coded for 256x256 output." << std::endl;
        }

        if (!vk_processor->Initialize(compute_shader_path,
                                      session->input_width,
                                      session->input_height)) {
            std::cerr << "Failed to initialize VulkanImageProcessor." << std::endl;
            return nullptr;
        }
        session->processor = vk_processor.release();

        if (session->input_channels == 3) {
            std::cout << "[Debug SuperRes_Initialize_GPU] Model needs 3 channels, Vulkan outputs 4. Creating 4-channel temp buffer." << std::endl;
            session->vulkan_temp_buffer.resize(session->input_width * session->input_height * 4);
        }
    }
    // ---------------------------------------------------------

    session->model = std::make_unique<litert::Model>(std::move(model));

    // --- THIS IS THE KEY DIFFERENCE ---
    // Call CreateGpuOptions() instead of CreateCpuOptions()
    litert::Options options = CreateGpuOptions(); 
    // ----------------------------------

    // --- NEW: Enable profiling in options ---
    LITERT_ASSIGN_OR_ABORT(auto runtime_options, litert::RuntimeOptions::Create());
    runtime_options.SetEnableProfiling(/*enabled=*/true);
    options.AddOpaqueOptions(std::move(runtime_options));
    // ---------------------------------------------

    LITERT_ASSIGN_OR_ABORT(auto compiled_model,
                           litert::CompiledModel::Create(*session->env, *session->model, options));
    session->compiled_model = std::make_unique<litert::CompiledModel>(std::move(compiled_model));

    LITERT_ASSIGN_OR_ABORT(auto input_buffers, session->compiled_model->CreateInputBuffers());
    session->input_buffers =
        std::make_unique<std::vector<litert::TensorBuffer>>(std::move(input_buffers));

    LITERT_ASSIGN_OR_ABORT(auto output_buffers, session->compiled_model->CreateOutputBuffers());
    session->output_buffers =
        std::make_unique<std::vector<litert::TensorBuffer>>(std::move(output_buffers));

    // --- REMOVED: OpenGL buffer creation ---
    // size_t buffer_size = ...
    // session->preprocessed_buffer_id = ...

    return session.release();  // Transfer ownership to the caller
}

// --- MODIFIED: SuperRes_Shutdown ---
void SuperRes_Shutdown(SuperResSession* session) {
    if (!session) return;

    // --- REMOVED: Cleanup GL resources ---
    // if (session->processor) { ... }
    
    // --- NEW: Clean up the pre-processor (from main_cpu.cc) ---
    if (session->processor) {
        if (session->preprocessor_type == kSuperResVulkanPreprocessor) {
            auto vk_processor = static_cast<VulkanImageProcessor*>(session->processor);
            vk_processor->Shutdown();
            delete vk_processor;
        }
    }
    // ---------------------------------------

    // Let smart pointers handle the rest of the cleanup
    delete session;
}

// --- REPLACED: SuperRes_PreProcess ---
// (Replaced entire function body with the one from main_cpu.cc)
bool SuperRes_PreProcess(SuperResSession* session, const ImageData* input_image) {
    if (!session || !input_image || !input_image->data) return false;

    if (session->preprocessor_type == kSuperResVulkanPreprocessor) {
        auto vk_processor = static_cast<VulkanImageProcessor*>(session->processor);
        
        float* vulkan_output_ptr = nullptr;
        bool needs_conversion = (session->input_channels == 3 && session->vulkan_temp_buffer.size() > 0);

        if (needs_conversion) {
            std::cout << "[Debug SuperRes_PreProcess_GPU] Writing Vulkan output to 4-channel temp buffer." << std::endl;
            vulkan_output_ptr = session->vulkan_temp_buffer.data();
        } else {
            std::cout << "[Debug SuperRes_PreProcess_GPU] Writing Vulkan output directly to model input buffer." << std::endl;
            vulkan_output_ptr = session->preprocessed_data.data();
        }

        if (!vk_processor->PreprocessImage(
                input_image->data,
                input_image->width,
                input_image->height,
                input_image->channels,
                vulkan_output_ptr)) {
            std::cerr << "VulkanImageProcessor::PreprocessImage failed." << std::endl;
            return false;
        }

        if (needs_conversion) {
            std::cout << "[Debug SuperRes_PreProcess_GPU] Converting 4-channel Vulkan output to 3-channel model input." << std::endl;
            int num_pixels = session->input_width * session->input_height;
            for (int i = 0; i < num_pixels; ++i) {
                session->preprocessed_data[i * 3 + 0] = session->vulkan_temp_buffer[i * 4 + 0]; // R
                session->preprocessed_data[i * 3 + 1] = session->vulkan_temp_buffer[i * 4 + 1]; // G
                session->preprocessed_data[i * 3 + 2] = session->vulkan_temp_buffer[i * 4 + 2]; // B
            }
        }

    } else {
        // --- Fallback to original CPU pre-processing ---
        std::cout << "[Debug SuperRes_PreProcess_GPU] Using CPU Pre-processor (ResizeImageBilinear)." << std::endl;
        ImageUtils::ResizeImageBilinear(
            input_image->data,
            input_image->width,
            input_image->height,
            input_image->channels,
            session->preprocessed_data.data(), // float* destination
            session->input_width,              // target width
            session->input_height,             // target height
            session->input_channels            // target channels
        );
    }

    // Write to the LiteRT input buffer
    std::cout << "[Debug SuperRes_PreProcess_GPU] Writing final data to TFLite input buffer." << std::endl;
    LITERT_ABORT_IF_ERROR(
        (*session->input_buffers)[0].Write(absl::MakeConstSpan(session->preprocessed_data)));

    return true;
}

// --- NEW: Added SuperRes_PreProcess_AHB (from main_cpu.cc) ---
#ifdef __ANDROID__
bool SuperRes_PreProcess_AHB(SuperResSession* session,
                             AHardwareBuffer* in_buffer,
                             int in_width,
                             int in_height) {
    if (!session || !in_buffer) return false;
    
    if (session->preprocessor_type != kSuperResVulkanPreprocessor) {
        std::cerr << "AHardwareBuffer input is only supported with the Vulkan preprocessor." << std::endl;
        return false;
    }

    auto vk_processor = static_cast<VulkanImageProcessor*>(session->processor);
    
    float* vulkan_output_ptr = nullptr;
    bool needs_conversion = (session->input_channels == 3 && session->vulkan_temp_buffer.size() > 0);

    if (needs_conversion) {
        std::cout << "[Debug SuperRes_PreProcess_AHB_GPU] Writing Vulkan output to 4-channel temp buffer." << std::endl;
        vulkan_output_ptr = session->vulkan_temp_buffer.data();
    } else {
        std::cout << "[Debug SuperRes_PreProcess_AHB_GPU] Writing Vulkan output directly to model input buffer." << std::endl;
        vulkan_output_ptr = session->preprocessed_data.data();
    }

    if (!vk_processor->PreprocessImage(
            in_buffer,
            in_width,
            in_height,
            vulkan_output_ptr)) {
        std::cerr << "VulkanImageProcessor::PreprocessImage (AHB) failed." << std::endl;
        return false;
    }

    if (needs_conversion) {
        std::cout << "[Debug SuperRes_PreProcess_AHB_GPU] Converting 4-channel Vulkan output to 3-channel model input." << std::endl;
        int num_pixels = session->input_width * session->input_height;
        for (int i = 0; i < num_pixels; ++i) {
            session->preprocessed_data[i * 3 + 0] = session->vulkan_temp_buffer[i * 4 + 0]; // R
            session->preprocessed_data[i * 3 + 1] = session->vulkan_temp_buffer[i * 4 + 1]; // G
            session->preprocessed_data[i * 3 + 2] = session->vulkan_temp_buffer[i * 4 + 2]; // B
        }
    }

    std::cout << "[Debug SuperRes_PreProcess_AHB_GPU] Writing final data to TFLite input buffer." << std::endl;
    LITERT_ABORT_IF_ERROR(
        (*session->input_buffers)[0].Write(absl::MakeConstSpan(session->preprocessed_data)));

    return true;
}
#endif
// -------------------------------------------------

// --- NEW: Added SuperRes_GetPreprocessedData (from main_cpu.cc) ---
const float* SuperRes_GetPreprocessedData(SuperResSession* session, 
                                        int* width, int* height, int* channels) {
    if (!session) return nullptr;
    if (width) *width = session->input_width;
    if (height) *height = session->input_height;
    if (channels) *channels = session->input_channels;
    
    if (session->preprocessed_data.empty()) {
        std::cerr << "Preprocessed data is empty. Call SuperRes_PreProcess first." << std::endl;
        return nullptr;
    }
    
    return session->preprocessed_data.data();
}
// -------------------------------------------

// --- MODIFIED: SuperRes_Run (to add profiling) ---
bool SuperRes_Run(SuperResSession* session) {
    if (!session) return false;

    // --- PROFILER: Get profiler from model ---
    LITERT_ASSIGN_OR_ABORT(auto profiler, session->compiled_model->GetProfiler());
    if (!profiler) {
        std::cerr << "Failed to get profiler." << std::endl;
    } else {
        if (!profiler.StartProfiling()) {
            std::cerr << "Failed to start profiling." << std::endl;
        }
    }
    // -----------------------------------------

    bool async = true;
    LITERT_ABORT_IF_ERROR(
        session->compiled_model->RunAsync(0, *session->input_buffers, *session->output_buffers, async));

    // --- PROFILER (ms): Get and print events in milliseconds ---
    if (profiler) {
        LITERT_ASSIGN_OR_ABORT(auto events, profiler.GetEvents());

        double allocate_tensors_ms = 0.0;
        double invoke_ms = 0.0;
        double other_ms = 0.0;
        double total_run_ms = 0.0;

        std::cout << "\n--- All Profiler Events (GPU Backend) ---" << std::endl;
        for (const auto& event : events) {
          std::cout << "Event Tag: " << event.tag
                    << ", Start (ms): " << (event.start_timestamp_us / 1000.0)
                    << ", Elapsed (ms): " << (event.elapsed_time_us / 1000.0) << std::endl;
        }
        std::cout << "---------------------------------------\n" << std::endl;


        // Calculate and print the summary
        for (const auto& event : events) {
            std::string tag(event.tag);
            double elapsed_ms = event.elapsed_time_us / 1000.0;

            if (tag == "AllocateTensors") {
                allocate_tensors_ms += elapsed_ms;
            } else if (tag == "Invoke") {
                if (event.start_timestamp_us > 0) {
                    invoke_ms += elapsed_ms;
                }
            } else if (tag == "LiteRT::Run[buffer registration]" || tag == "LiteRT::Run[Buffer sync]") {
                other_ms += elapsed_ms;
            }
        }

        total_run_ms = allocate_tensors_ms + invoke_ms + other_ms;

        std::cout << "--- Full Runtime Breakdown (GPU Backend) ---" << std::endl;
        std::cout << "AllocateTensors: " << allocate_tensors_ms << " ms" << std::endl;
        std::cout << "Invoke (Inference): " << invoke_ms << " ms" << std::endl;
        std::cout << "Other (Buffer sync/registration): " << other_ms << " ms" << std::endl;
        std::cout << "--------------------------------" << std::endl;
        std::cout << "Total time for Run call: " << total_run_ms << " ms" << std::endl;
        std::cout << "--------------------------------\n" << std::endl;
        
        if (!profiler.Reset()) {
            std::cerr << "Failed to reset profiler." << std::endl;
        }
    }
    // ----------------------------------------------------------

    return true;
}

// --- UNCHANGED: SuperRes_PostProcess ---
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

// --- UNCHANGED: SuperRes_FreeOutputData ---
void SuperRes_FreeOutputData(OutputData* output_data) {
    if (output_data && output_data->data) {
        delete[] output_data->data;
        output_data->data = nullptr;
    }
}

}  // extern "C"