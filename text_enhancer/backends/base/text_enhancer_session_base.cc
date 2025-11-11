#include <cstdint>
#include <string>

#include "absl/log/log.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_profiler.h"
#include "litert/cc/options/litert_runtime_options.h"

#include "text_enhancer_session_base.h"
#include "text_enhancer/utils/image_utils.h"

#ifdef __ANDROID__
#include "android/hardware_buffer.h"
#endif

// --- Initialize_Base, Run_Base ---
// [OMITTED FOR BREVITY - MODIFIED INITIALIZE_BASE BELOW]
TextEnhancerSession* TextEnhancer_Initialize_Base(const TextEnhancerOptions& options,
                                                  litert::Options litert_options,
                                                  std::unique_ptr<litert::Environment> env) {
    auto session = std::make_unique<TextEnhancerSession>();
    session->original_input_width = options.input_width;
    session->original_input_height = options.input_height;
    if (session->original_input_width == 0 || session->original_input_height == 0) {
        LOG(ERROR) << "Error: input_width and input_height must be set in "
                      "TextEnhancerOptions.";
        return nullptr;
    }
    if (options.compute_shader_path && std::string(options.compute_shader_path) != "") {
        session->preprocessor_type = TextEnhancerSession::PreprocessorType::kVulkan;
    } else {
        session->preprocessor_type = TextEnhancerSession::PreprocessorType::kCpu;
    }
    session->env = std::move(env);
    LITERT_ASSIGN_OR_ABORT(auto model, litert::Model::CreateFromFile(options.model_path));
    LITERT_ASSIGN_OR_ABORT(auto input_tensor_type, model.GetInputTensorType(0, 0));
    session->model_input_height = input_tensor_type.Layout().Dimensions()[1];
    session->model_input_width = input_tensor_type.Layout().Dimensions()[2];
    session->model_input_channels = input_tensor_type.Layout().Dimensions()[3];
    LITERT_ASSIGN_OR_ABORT(auto output_tensor_type, model.GetOutputTensorType(0, 0));
    session->model_output_height = output_tensor_type.Layout().Dimensions()[1];
    session->model_output_width = output_tensor_type.Layout().Dimensions()[2];
    session->model_output_channels = output_tensor_type.Layout().Dimensions()[3];
    LOG(INFO) << "[Debug TextEnhancer_Initialize] Model Input: " << session->model_input_width
              << "x" << session->model_input_height << "x" << session->model_input_channels;
    size_t num_elements =
        session->model_input_width * session->model_input_height * session->model_input_channels;

    // --- MODIFIED: Allocate double-buffers ---
    if (options.use_int8_preprocessor) {
        LOG(INFO) << "Model input type is Int8. Allocating uint8_t double buffer.";
        session->is_int8_input = true;
        for (int i = 0; i < session->kMaxFramesInFlight; ++i) {
            session->preprocessed_data_uint8_[i].resize(num_elements);
        }
    } else {
        LOG(INFO) << "Model input type is Float32. Allocating float double buffer.";
        session->is_int8_input = false;
        for (int i = 0; i < session->kMaxFramesInFlight; ++i) {
            session->preprocessed_data_float_[i].resize(num_elements);
        }
    }
    // --- END MODIFIED ---

    if (session->preprocessor_type == TextEnhancerSession::PreprocessorType::kVulkan) {
        LOG(INFO) << "Initializing Vulkan Pre-processor...";
        session->vulkan_processor = std::make_unique<VulkanImageProcessor>();
        if (options.use_int8_preprocessor != session->is_int8_input) {
            LOG(ERROR) << "Mismatch: use_int8_preprocessor option ("
                       << options.use_int8_preprocessor
                       << ") does not match model's actual input type (is_int8: "
                       << session->is_int8_input << ").";
            return nullptr;
        }
        const int kMaxInputChannels = 4;
        if (!session->vulkan_processor->Initialize(
                options.compute_shader_path,
                session->original_input_width,
                session->original_input_height,
                kMaxInputChannels,
                session->model_input_width,
                session->model_input_height,
                session->is_int8_input)) {
            LOG(ERROR) << "Failed to initialize VulkanImageProcessor.";
            return nullptr;
        }
    } else {
        LOG(INFO) << "Using CPU Pre-processor.";
        if (session->is_int8_input) {
            LOG(ERROR) << "CPU Pre-processor does not support Int8 output. "
                       << "Only Vulkan pre-processor does.";
            return nullptr;
        }
    }
    session->model = std::make_unique<litert::Model>(std::move(model));
    LITERT_ASSIGN_OR_ABORT(auto runtime_options, litert::RuntimeOptions::Create());
    runtime_options.SetEnableProfiling(true);
    litert_options.AddOpaqueOptions(std::move(runtime_options));
    LITERT_ASSIGN_OR_ABORT(
        auto compiled_model,
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
// [Run_Base is unchanged, OMITTED FOR BREVITY]
TextEnhancerStatus TextEnhancer_Run_Base(TextEnhancerSession* session, float* inference_time_ms,
                                         std::function<litert::Expected<void>()> run_fn) {
    if (!session) return kTextEnhancerInputError;
    LITERT_ASSIGN_OR_ABORT(auto profiler, session->compiled_model->GetProfiler());
    if (profiler) {
        profiler.StartProfiling();
    }
    auto run_status = run_fn();
    if (!run_status) {
        LOG(ERROR) << "CompiledModel::Run/RunAsync failed: " << run_status.Error().Message();
        return kTextEnhancerRuntimeError;
    }
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
            LOG(INFO) << "  Event " << event_index << ": [" << tag << "], Time: " << event_ms
                      << " ms";
            total_child_event_ms += event_ms;
            event_index++;
        }
        LOG(INFO) << "----------------------------------------";
        if (inference_time_ms) {
            *inference_time_ms = total_invoke_ms;
        }
        LOG(INFO) << "Total Child Event Time (Sum of Layers/Ops): " << total_child_event_ms
                  << " ms";
        LOG(INFO) << "Total 'Invoke' Time (Includes Overhead): " << total_invoke_ms << " ms";
        profiler.Reset();
    } else if (inference_time_ms) {
        *inference_time_ms = -1.0;
    }
    return kTextEnhancerOk;
}

// --- C API Implementation ---
extern "C" {

// --- Shutdown ---
// [OMITTED FOR BREVITY - NO CHANGES]
void TextEnhancer_Shutdown(TextEnhancerSession* session) {
    if (!session) return;
    delete session;
    LOG(INFO) << "TextEnhancer_Shutdown complete.";
}

// --- [REMOVED] TextEnhancer_PreProcess ---
// --- [REMOVED] TextEnhancer_PreProcess_AHB ---

// --- [ADDED] TextEnhancer_SubmitPreProcess ---
TextEnhancerStatus TextEnhancer_SubmitPreProcess(TextEnhancerSession* session,
                                                 const uint8_t* rgb_data) {
    if (!session || !rgb_data) return kTextEnhancerInputError;
    const int kInputChannels = 4;

    // Get the current buffer index for this frame
    int current_idx = session->frame_index_ % session->kMaxFramesInFlight;

    if (session->preprocessor_type == TextEnhancerSession::PreprocessorType::kVulkan) {
        if (!session->vulkan_processor) {
            LOG(ERROR) << "Vulkan preprocessor not initialized.";
            return kTextEnhancerFailed;
        }
        auto vk_processor = session->vulkan_processor.get();

        // Submit the work. This will wait for the buffer to be free (from 2 frames ago)
        // and then submit the new work, returning immediately.
        if (!vk_processor->SubmitPreprocessImage(rgb_data, session->original_input_width,
                                                 session->original_input_height, kInputChannels,
                                                 current_idx)) {
            LOG(ERROR) << "VulkanImageProcessor::SubmitPreprocessImage failed.";
            return kTextEnhancerRuntimeError;
        }
    } else {
        // CPU processing is synchronous, so "submit" and "sync" are the same.
        // We do the work here and it will be "synced" (i.e., ready) immediately.
        LOG(INFO) << "[Debug TextEnhancer_SubmitPreProcess] Using CPU Pre-processor.";
        ImageUtils::ResizeImageBilinear(
            rgb_data, session->original_input_width, session->original_input_height, kInputChannels,
            session->preprocessed_data_float_[current_idx].data(), // Use current buffer
            session->model_input_width, session->model_input_height,
            session->model_input_channels);
    }
    
    return kTextEnhancerOk;
}

#ifdef __ANDROID__
// --- [ADDED] TextEnhancer_SubmitPreProcess_AHB ---
TextEnhancerStatus TextEnhancer_SubmitPreProcess_AHB(TextEnhancerSession* session,
                                                     AHardwareBuffer* in_buffer) {
    if (!session || !in_buffer) return kTextEnhancerInputError;

    // Get the current buffer index for this frame
    int current_idx = session->frame_index_ % session->kMaxFramesInFlight;

    if (session->preprocessor_type != TextEnhancerSession::PreprocessorType::kVulkan) {
        LOG(ERROR) << "AHardwareBuffer input is only supported with the Vulkan "
                      "preprocessor.";
        return kTextEnhancerInputError;
    }
    if (!session->vulkan_processor) {
        LOG(ERROR) << "Vulkan preprocessor not initialized.";
        return kTextEnhancerFailed;
    }
    auto vk_processor = session->vulkan_processor.get();

    // Submit the work.
    if (!vk_processor->SubmitPreprocessImage(in_buffer, session->original_input_width,
                                             session->original_input_height, current_idx)) {
        LOG(ERROR) << "VulkanImageProcessor::SubmitPreprocessImage (AHB) failed.";
        return kTextEnhancerRuntimeError;
    }
    
    return kTextEnhancerOk;
}
#endif  // __ANDROID__

// --- [ADDED] TextEnhancer_SyncPreProcess ---
TextEnhancerStatus TextEnhancer_SyncPreProcess(TextEnhancerSession* session) {
    if (!session) return kTextEnhancerInputError;

    // Get the index for the frame that was just submitted
    int current_idx = session->frame_index_ % session->kMaxFramesInFlight;
    void* vulkan_output_ptr = nullptr;

    if (session->is_int8_input) {
        vulkan_output_ptr = session->preprocessed_data_uint8_[current_idx].data();
    } else {
        vulkan_output_ptr = session->preprocessed_data_float_[current_idx].data();
    }

    if (session->preprocessor_type == TextEnhancerSession::PreprocessorType::kVulkan) {
        auto vk_processor = session->vulkan_processor.get();
        
        // Wait for the submitted work to complete and copy the data
        // into our preprocessed_data buffer.
        if (!vk_processor->SyncPreprocess(vulkan_output_ptr, current_idx)) {
            LOG(ERROR) << "VulkanImageProcessor::SyncPreprocess failed.";
            return kTextEnhancerRuntimeError;
        }
        
        // Save the timings for this frame
        session->last_synced_vulkan_timings_ = vk_processor->GetLastTimings(current_idx);
    } else {
        // CPU work was already done in "Submit", so just "sync" timings.
        session->last_synced_vulkan_timings_ = {}; // Zero out timings
    }

    // Now that the data is ready, write it to the LiteRT input buffer
    litert::Expected<void> status;
    if (session->is_int8_input) {
        status = (*session->input_buffers)[0].Write(
            absl::MakeConstSpan(session->preprocessed_data_uint8_[current_idx]));
    } else {
        status = (*session->input_buffers)[0].Write(
            absl::MakeConstSpan(session->preprocessed_data_float_[current_idx]));
    }

    if (!status) {
        LOG(ERROR) << "Failed to write to input buffer: " << status.Error().Message();
        return kTextEnhancerRuntimeError;
    }

    // Increment the frame index to move to the next buffer
    session->frame_index_++;
    
    return kTextEnhancerOk;
}


// --- GetPreprocessedData, PostProcess, FreeOutputData ---
// [OMITTED FOR BREVITY - MODIFIED GetPreprocessedData BELOW]
TextEnhancerStatus TextEnhancer_GetPreprocessedData(TextEnhancerSession* session, uint8_t** data) {
    if (!session || !data) return kTextEnhancerInputError;

    // --- MODIFIED: Return the *last synced* frame's data ---
    // frame_index_ was already incremented by Sync.
    // So the last synced index is (frame_index_ - 1).
    int synced_idx = (session->frame_index_ - 1) % session->kMaxFramesInFlight;

    if (session->is_int8_input) {
        if (session->preprocessed_data_uint8_[synced_idx].empty()) {
            LOG(ERROR) << "Preprocessed data (uint8) is empty. Call Submit/Sync first.";
            return kTextEnhancerFailed;
        }
        *data = session->preprocessed_data_uint8_[synced_idx].data();
    } else {
        if (session->preprocessed_data_float_[synced_idx].empty()) {
            LOG(ERROR) << "Preprocessed data (float) is empty. Call Submit/Sync first.";
            return kTextEnhancerFailed;
        }
        *data = reinterpret_cast<uint8_t*>(session->preprocessed_data_float_[synced_idx].data());
    }
    return kTextEnhancerOk;
}
// [PostProcess and FreeOutputData are unchanged, OMITTED FOR BREVITY]
TextEnhancerStatus TextEnhancer_PostProcess(TextEnhancerSession* session,
                                            TextEnhancerOutput& output) {
    if (!session) return kTextEnhancerInputError;
    output.data = nullptr;
    output.width = 0;
    output.height = 0;
    output.channels = 0;
#ifdef __ANDROID__
    output.output_buffer = nullptr;
#endif
    if ((*session->output_buffers)[0].HasEvent()) {
        LITERT_ASSIGN_OR_ABORT(auto event, (*session->output_buffers)[0].GetEvent());
        event.Wait();
    }
#ifdef __ANDROID__
    auto ahb_expected = (*session->output_buffers)[0].GetAhwb();
    if (ahb_expected.HasValue()) {
        LOG(INFO) << "PostProcess: Using AHardwareBuffer output path.";
        AHardwareBuffer* ahb = ahb_expected.Value();
        AHardwareBuffer_acquire(ahb);
        output.output_buffer = ahb;
        output.width = session->model_output_width;
        output.height = session->model_output_height;
        output.channels = session->model_output_channels;
        output.data = nullptr; 
        return kTextEnhancerOk;
    } else {
        LOG(WARNING) << "PostProcess: GetAhwb() failed or not supported ("
                     << ahb_expected.Error().Message() << "). Falling back to CPU read path.";
    }
#endif
    LOG(INFO) << "PostProcess: Using CPU fallback path (Read to vector).";
    size_t output_size =
        session->model_output_width * session->model_output_height * session->model_output_channels;
    if (session->is_int8_input) {
        std::vector<uint8_t> output_vec(output_size);
        auto read_status = (*session->output_buffers)[0].Read(absl::MakeSpan(output_vec));
        if (!read_status) {
            LOG(ERROR) << "Failed to read output buffer: " << read_status.Error().Message();
            return kTextEnhancerRuntimeError;
        }
        size_t output_bytes = output_size * sizeof(uint8_t);
        uint8_t* data_ptr = new (std::nothrow) uint8_t[output_bytes];
        if (!data_ptr) {
            LOG(ERROR) << "Failed to allocate memory for output data.";
            return kTextEnhancerFailed;
        }
        memcpy(data_ptr, output_vec.data(), output_size * sizeof(uint8_t));
        output.data = data_ptr;
    } else {
        std::vector<float> output_vec(output_size);
        auto read_status = (*session->output_buffers)[0].Read(absl::MakeSpan(output_vec));
        if (!read_status) {
            LOG(ERROR) << "Failed to read output buffer: " << read_status.Error().Message();
            return kTextEnhancerRuntimeError;
        }
        size_t output_bytes = output_size * sizeof(float);
        uint8_t* data_ptr = new (std::nothrow) uint8_t[output_bytes];
        if (!data_ptr) {
            LOG(ERROR) << "Failed to allocate memory for output data.";
            return kTextEnhancerFailed;
        }
        memcpy(data_ptr, output_vec.data(), output_size * sizeof(float));
        output.data = data_ptr;
    }
    output.width = session->model_output_width;
    output.height = session->model_output_height;
    output.channels = session->model_output_channels;
    return kTextEnhancerOk;
}
void TextEnhancer_FreeOutputData(TextEnhancerOutput& output) {
#ifdef __ANDROID__
    if (output.output_buffer) {
        AHardwareBuffer_release(output.output_buffer);
        output.output_buffer = nullptr;
    }
#endif
    if (output.data) {
        delete[] output.data;
        output.data = nullptr;
    }
    output.width = 0;
    output.height = 0;
    output.channels = 0;
}


/**
 * @brief Gets the detailed timings from the last pre-processing step.
 */
TextEnhancerStatus TextEnhancer_GetLastPreprocessorTimings(
    TextEnhancerSession* session,
    TextEnhancerPreprocessorTimings* timings) {
    if (!session || !timings) return kTextEnhancerInputError;

    // --- MODIFIED: Copy from the last *synced* frame's timings ---
    const auto& last_timings = session->last_synced_vulkan_timings_;

    timings->staging_copy_ms = last_timings.staging_copy_ms;
    timings->gpu_submit_wait_ms = last_timings.gpu_submit_wait_ms;
    timings->readback_copy_ms = last_timings.readback_copy_ms;
    timings->gpu_shader_ms = last_timings.gpu_shader_ms;
    timings->gpu_readback_ms = last_timings.gpu_readback_ms;
    // --- END MODIFIED ---
    
    return kTextEnhancerOk;
}


}  // extern "C"