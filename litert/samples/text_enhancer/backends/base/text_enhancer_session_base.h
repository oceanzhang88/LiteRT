#pragma once

#include <functional>  // For std::function
#include <memory>
#include <vector>

// --- LiteRT C++ API Headers ---
#include "absl/log/log.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
// ------------------------------

// --- Project Headers ---
#include "litert/samples/text_enhancer/image_processing/vulkan_image_processor.h"
#include "litert/samples/text_enhancer/text_enhancer_api.h"
// ---------------------

/**
 * @brief The internal implementation of the opaque TextEnhancerSession.
 * (This is the common structure shared by CPU, GPU, and NPU backends)
 */
struct TextEnhancerSession {
    // --- Internal preprocessor type ---
    enum class PreprocessorType { kCpu, kVulkan };
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
    std::unique_ptr<VulkanImageProcessor> vulkan_processor;
    std::vector<float> preprocessed_data;
    std::vector<float> vulkan_temp_buffer;
};

/**
 * @brief Base implementation for TextEnhancer_Initialize.
 * This performs all common initialization steps after the backend-specific
 * Environment and Options have been created.
 */
TextEnhancerSession* TextEnhancer_Initialize_Base(const TextEnhancerOptions& options,
                                                  litert::Options litert_options,
                                                  std::unique_ptr<litert::Environment> env);

/**
 * @brief Base implementation for TextEnhancer_Run.
 * This handles all common profiling logic and executes the backend-specific
 * run function provided via `run_fn`.
 */
TextEnhancerStatus TextEnhancer_Run_Base(TextEnhancerSession* session, float* inference_time_ms,
                                         std::function<litert::Expected<void>()> run_fn);