#include <utility>  // For std::move

// --- Base Header ---
#include "base/text_enhancer_session_base.h"
// -------------------

// --- LiteRT Headers ---
#include "litert/cc/options/litert_cpu_options.h"
// --------------------

namespace {
// Creates LiteRT CPU options.
litert::Options CreateCpuOptions() {
    LITERT_ASSIGN_OR_ABORT(auto cpu_options, litert::CpuOptions::Create());
    LITERT_ABORT_IF_ERROR(cpu_options.SetNumThreads(4));

    LITERT_ASSIGN_OR_ABORT(litert::Options options, litert::Options::Create());
    options.AddOpaqueOptions(std::move(cpu_options));
    options.SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);
    return options;
}
}  // namespace

// --- C API Implementation ---
extern "C" {

/**
 * @brief Initializes the Text Enhancer instance (CPU Backend).
 */
TextEnhancerSession* TextEnhancer_Initialize(const TextEnhancerOptions& options) {
    LOG(INFO) << "TextEnhancer_Initialize (CPU Backend)...";

    // 1. Create backend-specific environment
    LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));
    auto env_ptr = std::make_unique<litert::Environment>(std::move(env));

    // 2. Create backend-specific options
    litert::Options litert_options = CreateCpuOptions();

    // 3. Call base initializer
    return TextEnhancer_Initialize_Base(options, std::move(litert_options), std::move(env_ptr));
}

/**
 * @brief Runs the inference (CPU Backend).
 */
TextEnhancerStatus TextEnhancer_Run(TextEnhancerSession* session, float* inference_time_ms) {
    // Create a std::function object that matches the base function's signature.
    // This replaces 'auto run_fn = ...'
    auto run_fn = [&]() -> litert::Expected<void> {
        return session->compiled_model->Run(*session->input_buffers, *session->output_buffers);
    };

    // Call the base run function to handle profiling
    return TextEnhancer_Run_Base(session, inference_time_ms, run_fn);
}

}  // extern "C"