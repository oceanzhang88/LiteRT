#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_options.h"

// --- MODIFIED: Include Vulkan processor header ---
#include "text_enhancer/image_processing/vulkan_image_processor.h"
#include "text_enhancer/text_enhancer_api.h"  // For TextEnhancerOptions

/**
 * @brief Opaque struct holding all session state.
 */
struct TextEnhancerSession {
    // --- Environment & Model ---
    std::unique_ptr<litert::Environment> env;
    std::unique_ptr<litert::Model> model;
    std::unique_ptr<litert::CompiledModel> compiled_model;
    std::unique_ptr<std::vector<litert::TensorBuffer>> input_buffers;
    std::unique_ptr<std::vector<litert::TensorBuffer>> output_buffers;

    // --- Pre-processor ---
    enum class PreprocessorType { kCpu, kVulkan };
    PreprocessorType preprocessor_type = PreprocessorType::kCpu;
    std::unique_ptr<VulkanImageProcessor> vulkan_processor;

    // --- Model/Image Dimensions ---
    int original_input_width = 0;
    int original_input_height = 0;
    int model_input_width = 0;
    int model_input_height = 0;
    int model_input_channels = 0;
    int model_output_width = 0;
    int model_output_height = 0;
    int model_output_channels = 0;

    // --- Pre-processed Data Buffers ---
    bool is_int8_input = false;
    std::vector<float> preprocessed_data_float;
    std::vector<uint8_t> preprocessed_data_uint8; // For int8 models

    // --- Member to store last Vulkan timings ---
    VulkanImageProcessor::TimingInfo last_vulkan_timings;
};

/**
 * @brief Base implementation for TextEnhancer_Initialize.
 */
TextEnhancerSession* TextEnhancer_Initialize_Base(const TextEnhancerOptions& options,
                                                  litert::Options litert_options,
                                                  std::unique_ptr<litert::Environment> env);

/**
 * @brief Base implementation for TextEnhancer_Run.
 */
TextEnhancerStatus TextEnhancer_Run_Base(TextEnhancerSession* session, float* inference_time_ms,
                                         std::function<litert::Expected<void>()> run_fn);