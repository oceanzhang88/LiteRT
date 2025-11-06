#include <utility> // For std::move

// --- Base Header ---
#include "base/text_enhancer_session_base.h"
// -------------------

// --- LiteRT Headers ---
#include "litert/cc/options/litert_gpu_options.h"
// --------------------

namespace {
// --- MODIFIED: Creates LiteRT GPU options (from original gpu.cc) ---
litert::Options CreateGpuOptions() {
    // Hardcode use_gl_buffers = false, as this is no longer passed by the API
    // bool use_gl_buffers = false;

    LITERT_ASSIGN_OR_ABORT(auto gpu_options, litert::GpuOptions::Create());
    LITERT_ABORT_IF_ERROR(
        gpu_options.SetGpuBackend(kLiteRtGpuBackendOpenCl));  // Android
                                                              // kLiteRtGpuBackendOpenCl

    LITERT_ABORT_IF_ERROR(gpu_options.EnableExternalTensorsMode(true));

    LITERT_ASSIGN_OR_ABORT(litert::Options options, litert::Options::Create());
    options.SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
    options.AddOpaqueOptions(std::move(gpu_options));
    return options;
}
}  // namespace

// --- C API Implementation ---
extern "C" {

/**
 * @brief Initializes the Text Enhancer instance (GPU Backend).
 */
TextEnhancerSession* TextEnhancer_Initialize(
    const TextEnhancerOptions& options) {
    LOG(INFO) << "TextEnhancer_Initialize (GPU Backend)...";

    // 1. Create backend-specific environment
    LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));
    auto env_ptr = std::make_unique<litert::Environment>(std::move(env));

    // 2. Create backend-specific options
    litert::Options litert_options = CreateGpuOptions();

    // 3. Call base initializer
    return TextEnhancer_Initialize_Base(options, std::move(litert_options),
                                        std::move(env_ptr));
}

/**
 * @brief Runs the inference (GPU Backend).
 */
TextEnhancerStatus TextEnhancer_Run(TextEnhancerSession* session,
                                    float* inference_time_ms) {
    // Create a lambda for the backend-specific run call
    auto run_fn = [&]() {
        bool async = true;
        return session->compiled_model->RunAsync(0, *session->input_buffers,
                                                 *session->output_buffers, async);
    };

    // Call the base run function to handle profiling
    return TextEnhancer_Run_Base(session, inference_time_ms, run_fn);
}

}  // extern "C"