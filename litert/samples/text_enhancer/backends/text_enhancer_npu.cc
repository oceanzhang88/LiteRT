#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <utility> // For std::move

// --- LiteRT C++ API Headers ---
#include "absl/types/span.h"
// --- NEW: Include Abseil Logging ---
#include "absl/log/log.h"
// ------------------------------------
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
// --- MODIFIED: Include NPU Options ---
#include "litert/cc/options/litert_qualcomm_options.h"
// -------------------------------------
#include "litert/cc/litert_profiler.h"
#include "litert/cc/options/litert_runtime_options.h"
// ------------------------------

// --- Project Headers ---
#include "litert/samples/text_enhancer/text_enhancer_api.h"
#include "litert/samples/text_enhancer/utils/image_utils.h"
#include "litert/samples/text_enhancer/image_processing/vulkan_image_processor.h"
// ---------------------

#ifdef __ANDROID__
#include "android/hardware_buffer.h"
#endif

namespace {
// --- MODIFIED: Creates LiteRT NPU options (from original npu.cc) ---
litert::Options CreateNpuOptions() {
    LITERT_ASSIGN_OR_ABORT(auto qnn_options, litert::qualcomm::QualcommOptions::Create());
    // Add any NPU-specific options here if needed
    qnn_options.SetHtpPerformanceMode(kLiteRtQualcommHtpPerformanceModeBurst);
    qnn_options.SetUseFoldReLU(true);
    qnn_options.SetUseConvHMX(true);
    qnn_options.SetUseHtpPreference(true);
    qnn_options.SetOptimizationLevel(kHtpOptimizeForInferenceO3);

    LITERT_ASSIGN_OR_ABORT(litert::Options options, litert::Options::Create());
    options.SetHardwareAccelerators(kLiteRtHwAcceleratorNpu);
    options.AddOpaqueOptions(std::move(qnn_options));
    return options;
}
}  // namespace

/**
 * @brief The internal implementation of the opaque TextEnhancerSession.
 * (Copied from text_enhancer_cpu.cc)
 */
struct TextEnhancerSession {
    // --- Internal preprocessor type ---
    enum class PreprocessorType {
        kCpu,
        kVulkan
    };
    PreprocessorType preprocessor_type;

    // --- C++ LiteRT objects ---
    std::unique_ptr<litert::Environment> env;
    std::unique_ptr<litert::Model> model;
    std::unique_ptr<litert::CompiledModel> compiled_model;
    std::unique_ptr<std::vector<litert::TensorBuffer>> input_buffers;
    std::unique_ptr<std::vector<litert::TensorBuffer>> output_buffers;

    // --- Original image dimensions (from TextEnhancerOptions) ---
    int original_input_width = 0;
    int original_input_height = 0;

    // --- Model dimensions (from the .tflite file) ---
    int model_input_width = 0;
    int model_input_height = 0;
    int model_input_channels = 0;
    int model_output_width = 0;
    int model_output_height = 0;
    int model_output_channels = 0;

    // --- Pre-processing resources ---
    std::unique_ptr<VulkanImageProcessor> vulkan_processor; // Will not be initialized
    std::vector<float> preprocessed_data;
    std::vector<float> vulkan_temp_buffer; // Will not be used
};

// --- C API Implementation ---
extern "C" {

/**
 * @brief Initializes the Text Enhancer instance.
 */
TextEnhancerSession* TextEnhancer_Initialize(const TextEnhancerOptions& options) {
    // --- UPDATED: Use Abseil Log ---
    LOG(INFO) << "TextEnhancer_Initialize (NPU Backend)...";
    auto session = std::make_unique<TextEnhancerSession>();

    session->original_input_width = options.input_width;
    session->original_input_height = options.input_height;
    if (session->original_input_width == 0 || session->original_input_height == 0) {
        LOG(ERROR) << "Error: input_width and input_height must be set in TextEnhancerOptions.";
        return nullptr;
    }

    if (options.compute_shader_path && std::string(options.compute_shader_path) != "") {
        session->preprocessor_type = TextEnhancerSession::PreprocessorType::kVulkan;
    } else {
        session->preprocessor_type = TextEnhancerSession::PreprocessorType::kCpu;
    }

    // // --- NPU MODIFICATION: NPU backend does not support Vulkan pre-processing ---
    // if (options.compute_shader_path && std::string(options.compute_shader_path) != "") {
    //     LOG(WARNING) << "NPU backend does not support Vulkan pre-processing. "
    //                  << "Ignoring compute_shader_path and forcing CPU pre-processor.";
    // }
    // session->preprocessor_type = TextEnhancerSession::PreprocessorType::kCpu;
    // // --------------------------------------------------------------------------

    // --- NPU MODIFICATION: Add environment options from original npu.cc ---
    std::vector<litert::Environment::Option> environment_options;
    environment_options.push_back(litert::Environment::Option{
        litert::Environment::OptionTag::DispatchLibraryDir,
        absl::string_view("/data/local/tmp/super_res_acc_android/npu/"),
    });
    LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create(std::move(environment_options)));
    // ----------------------------------------------------------------------
    session->env = std::make_unique<litert::Environment>(std::move(env));

    LITERT_ASSIGN_OR_ABORT(auto model, litert::Model::CreateFromFile(options.model_path));

    LITERT_ASSIGN_OR_ABORT(auto input_tensor_type, model.GetInputTensorType(0, 0));
    session->model_input_height = input_tensor_type.Layout().Dimensions()[1];
    session->model_input_width = input_tensor_type.Layout().Dimensions()[2];
    session->model_input_channels = input_tensor_type.Layout().Dimensions()[3];

    LITERT_ASSIGN_OR_ABORT(auto output_tensor_type, model.GetOutputTensorType(0, 0));
    session->model_output_height = output_tensor_type.Layout().Dimensions()[1];
    session->model_output_width = output_tensor_type.Layout().Dimensions()[2];
    session->model_output_channels = output_tensor_type.Layout().Dimensions()[3];

    LOG(INFO) << "[Debug TextEnhancer_Initialize] Model Input: " 
              << session->model_input_width << "x" 
              << session->model_input_height << "x" 
              << session->model_input_channels;

    session->preprocessed_data.resize(session->model_input_width * session->model_input_height *
                                     session->model_input_channels);

    // --- NPU MODIFICATION: This block will be skipped as preprocessor_type is kCpu ---
    if (session->preprocessor_type == TextEnhancerSession::PreprocessorType::kVulkan) {
        LOG(INFO) << "Initializing Vulkan Pre-processor...";
        session->vulkan_processor = std::make_unique<VulkanImageProcessor>();

        if (!session->vulkan_processor->Initialize(options.compute_shader_path,
                                                 session->model_input_width,
                                                 session->model_input_height))
        {
            LOG(ERROR) << "Failed to initialize VulkanImageProcessor.";
            return nullptr;
        }

        if (session->model_input_channels == 3) {
            LOG(INFO) << "[Debug TextEnhancer_Initialize] Model needs 3ch, Vulkan outputs 4ch. Creating 4ch temp buffer.";
            session->vulkan_temp_buffer.resize(session->model_input_width * session->model_input_height * 4);
        }
    } else {
        LOG(INFO) << "Using CPU Pre-processor.";
    }
    // -------------------------------------------------------------------------------

    session->model = std::make_unique<litert::Model>(std::move(model));

    // --- MODIFIED: Call CreateNpuOptions() ---
    litert::Options litert_options = CreateNpuOptions();
    // -----------------------------------------

    LITERT_ASSIGN_OR_ABORT(auto runtime_options, litert::RuntimeOptions::Create());
    runtime_options.SetEnableProfiling(true);
    litert_options.AddOpaqueOptions(std::move(runtime_options));

    LITERT_ASSIGN_OR_ABORT(auto compiled_model,
                           litert::CompiledModel::Create(*session->env, *session->model, litert_options));
    session->compiled_model = std::make_unique<litert::CompiledModel>(std::move(compiled_model));

    LITERT_ASSIGN_OR_ABORT(auto input_buffers, session->compiled_model->CreateInputBuffers());
    session->input_buffers =
        std::make_unique<std::vector<litert::TensorBuffer>>(std::move(input_buffers));

    LITERT_ASSIGN_OR_ABORT(auto output_buffers, session->compiled_model->CreateOutputBuffers());
    session->output_buffers =
        std::make_unique<std::vector<litert::TensorBuffer>>(std::move(output_buffers));

    return session.release();
}

/**
 * @brief Shuts down the instance.
 */
void TextEnhancer_Shutdown(TextEnhancerSession* session) {
    if (!session) return;
    delete session;
    LOG(INFO) << "TextEnhancer_Shutdown complete.";
}

/**
 * @brief Pre-processes a raw CPU buffer.
 * (This function is identical to the CPU version, but will always take the
 * 'else' path for CPU pre-processing.)
 */
TextEnhancerStatus TextEnhancer_PreProcess(TextEnhancerSession* session, const uint8_t* rgb_data) {
    if (!session || !rgb_data) return kTextEnhancerInputError;
    
    const int kInputChannels = 4; 

    if (session->preprocessor_type == TextEnhancerSession::PreprocessorType::kVulkan) {
        // This path will not be taken on NPU backend
        LOG(ERROR) << "NPU backend incorrectly selected Vulkan preprocessor.";
        return kTextEnhancerFailed;

    } else {
        // --- CPU Pre-processing ---
        LOG(INFO) << "[Debug TextEnhancer_PreProcess] Using CPU Pre-processor (ResizeImageBilinear).";
        ImageUtils::ResizeImageBilinear(
            rgb_data,
            session->original_input_width,
            session->original_input_height,
            kInputChannels,
            session->preprocessed_data.data(),
            session->model_input_width,
            session->model_input_height,
            session->model_input_channels
        );
    }

    auto status = (*session->input_buffers)[0].Write(absl::MakeConstSpan(session->preprocessed_data));
    if (!status) {
        LOG(ERROR) << "Failed to write to input buffer: " << status.Error().Message();
        return kTextEnhancerRuntimeError;
    }

    return kTextEnhancerOk;
}

#ifdef __ANDROID__
/**
 * @brief Pre-processes an AHardwareBuffer.
 */
TextEnhancerStatus TextEnhancer_PreProcess_AHB(TextEnhancerSession* session, AHardwareBuffer* in_buffer) {
    if (!session || !in_buffer) return kTextEnhancerInputError;
    
    if (session->preprocessor_type != TextEnhancerSession::PreprocessorType::kVulkan) {
        LOG(ERROR) << "AHardwareBuffer input is only supported with the Vulkan preprocessor.";
        return kTextEnhancerInputError;
    }
    if (!session->vulkan_processor) {
        LOG(ERROR) << "Vulkan preprocessor not initialized.";
        return kTextEnhancerFailed;
    }

    auto vk_processor = session->vulkan_processor.get();
    
    float* vulkan_output_ptr = nullptr;
    bool needs_conversion = (session->model_input_channels == 3 && session->vulkan_temp_buffer.size() > 0);

    if (needs_conversion) {
        vulkan_output_ptr = session->vulkan_temp_buffer.data();
    } else {
        vulkan_output_ptr = session->preprocessed_data.data();
    }

    if (!vk_processor->PreprocessImage(
            in_buffer,
            session->original_input_width,
            session->original_input_height,
            vulkan_output_ptr)) {
        LOG(ERROR) << "VulkanImageProcessor::PreprocessImage (AHB) failed.";
        return kTextEnhancerRuntimeError;
    }

    if (needs_conversion) {
        int num_pixels = session->model_input_width * session->model_input_height;
        for (int i = 0; i < num_pixels; ++i) {
            session->preprocessed_data[i * 3 + 0] = session->vulkan_temp_buffer[i * 4 + 0]; // R
            session->preprocessed_data[i * 3 + 1] = session->vulkan_temp_buffer[i * 4 + 1]; // G
            session->preprocessed_data[i * 3 + 2] = session->vulkan_temp_buffer[i * 4 + 2]; // B
        }
    }

    auto status = (*session->input_buffers)[0].Write(absl::MakeConstSpan(session->preprocessed_data));
    if (!status) {
        LOG(ERROR) << "Failed to write to input buffer: " << status.Error().Message();
        return kTextEnhancerRuntimeError;
    }

    return kTextEnhancerOk;
}
#endif // __ANDROID__

/**
 * @brief Gets the preprocessed data buffer.
 */
TextEnhancerStatus TextEnhancer_GetPreprocessedData(TextEnhancerSession* session, uint8_t** data) {
    if (!session || !data) return kTextEnhancerInputError;
    
    if (session->preprocessed_data.empty()) {
        LOG(ERROR) << "Preprocessed data is empty. Call TextEnhancer_PreProcess first.";
        return kTextEnhancerFailed;
    }
    
    *data = reinterpret_cast<uint8_t*>(session->preprocessed_data.data());
    
    return kTextEnhancerOk;
}

/**
 * @brief Runs the inference.
 * (Identical to CPU backend, including profiling)
 */
TextEnhancerStatus TextEnhancer_Run(TextEnhancerSession* session, float* inference_time_ms) {
    if (!session) return kTextEnhancerInputError;

    LITERT_ASSIGN_OR_ABORT(auto profiler, session->compiled_model->GetProfiler());
    if (profiler) {
        profiler.StartProfiling();
    }

    // --- Run inference ---
    auto run_status = session->compiled_model->Run(*session->input_buffers, *session->output_buffers);
    if (!run_status) {
        LOG(ERROR) << "CompiledModel::Run failed: " << run_status.Error().Message();
        return kTextEnhancerRuntimeError;
    }

    // --- Profiler Event Processing ---
    if (profiler) {
        LITERT_ASSIGN_OR_ABORT(auto events, profiler.GetEvents());
        
        double total_invoke_ms = 0.0;
        double total_child_event_ms = 0.0;
        int event_index = 0;

        for (const auto& event : events) {
            if (std::string(event.tag) == "Invoke" && event.start_timestamp_us > 0) {
                total_invoke_ms = event.elapsed_time_us / 1000.0;
                break;
            }
        }

        LOG(INFO) << "--- Per-Event Profiler Breakdown ---";
        for (const auto& event : events) {
            std::string tag(event.tag);
            double event_ms = event.elapsed_time_us / 1000.0;

            // if (tag != "Invoke" && event.start_timestamp_us > 0) {
                LOG(INFO) << "  Event " << event_index << ": [" << tag << "], Time: " << event_ms << " ms";
                total_child_event_ms += event_ms;
                event_index++;
            // }
        }
        LOG(INFO) << "----------------------------------------";
        
        if (inference_time_ms) {
            *inference_time_ms = total_invoke_ms;
        }

        LOG(INFO) << "Total Child Event Time (Sum of Layers/Ops): " << total_child_event_ms << " ms";
        LOG(INFO) << "Total 'Invoke' Time (Includes Overhead): " << total_invoke_ms << " ms";

        profiler.Reset();
    } else if (inference_time_ms) {
        *inference_time_ms = -1.0; 
    }
    // --- END: Profiler Event Processing ---

    return kTextEnhancerOk;
}

/**
 * @brief Gets the output data.
 */
TextEnhancerStatus TextEnhancer_PostProcess(TextEnhancerSession* session, TextEnhancerOutput& output) {
    if (!session) return kTextEnhancerInputError;

    if ((*session->output_buffers)[0].HasEvent()) {
        LITERT_ASSIGN_OR_ABORT(auto event, (*session->output_buffers)[0].GetEvent());
        event.Wait();
    }

    size_t output_size = session->model_output_width * session->model_output_height * session->model_output_channels;

    std::vector<float> output_vec(output_size);
    auto read_status = (*session->output_buffers)[0].Read(absl::MakeSpan(output_vec));
    if (!read_status) {
        LOG(ERROR) << "Failed to read output buffer: " << read_status.Error().Message();
        return kTextEnhancerRuntimeError;
    }

    float* data_ptr = new (std::nothrow) float[output_size];
    if (!data_ptr) {
        LOG(ERROR) << "Failed to allocate memory for output data.";
        return kTextEnhancerFailed;
    }

    memcpy(data_ptr, output_vec.data(), output_size * sizeof(float));

    output.width = session->model_output_width;
    output.height = session->model_output_height;
    output.channels = session->model_output_channels;
    output.data = reinterpret_cast<uint8_t*>(data_ptr);

    return kTextEnhancerOk;
}

/**
 * @brief Frees the output data buffer.
 */
void TextEnhancer_FreeOutputData(TextEnhancerOutput& output) {
    if (output.data) {
        float* data_to_free = reinterpret_cast<float*>(output.data);
        delete[] data_to_free;
        output.data = nullptr;
        output.width = 0;
        output.height = 0;
        output.channels = 0;
    }
}


}  // extern "C"
