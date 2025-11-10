#include <utility>

// --- Base Header ---
#include "base/text_enhancer_session_base.h"
// -------------------

// --- LiteRT Headers ---
#include "litert/cc/options/litert_qualcomm_options.h"
// --------------------

namespace {
// --- MODIFIED: Creates LiteRT NPU options (from original npu.cc) ---
litert::Options CreateNpuOptions() {
    LITERT_ASSIGN_OR_ABORT(auto qnn_options, litert::qualcomm::QualcommOptions::Create());
    // Add any NPU-specific options here if needed
    qnn_options.SetHtpPerformanceMode(kLiteRtQualcommHtpPerformanceModeBurst);
    qnn_options.SetUseFoldReLU(true);
    qnn_options.SetUseConvHMX(true);
    qnn_options.SetNumHvxThreads(4);
    qnn_options.SetUseHtpPreference(true);
    qnn_options.SetOptimizationLevel(kHtpOptimizeForInferenceO3);

    LITERT_ASSIGN_OR_ABORT(litert::Options options, litert::Options::Create());
    options.SetHardwareAccelerators(kLiteRtHwAcceleratorNpu);
    options.AddOpaqueOptions(std::move(qnn_options));
    return options;
}
}  // namespace

// --- C API Implementation ---
extern "C" {

/**
 * @brief Initializes the Text Enhancer instance (NPU Backend).
 */
TextEnhancerSession* TextEnhancer_Initialize(const TextEnhancerOptions& options) {
    LOG(INFO) << "TextEnhancer_Initialize (NPU Backend)...";

    // 1. Create backend-specific environment
    std::vector<litert::Environment::Option> environment_options;
    environment_options.push_back(litert::Environment::Option{
        litert::Environment::OptionTag::DispatchLibraryDir,
        absl::string_view("/data/local/tmp/super_res_acc_android/npu/"),
    });
    LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create(std::move(environment_options)));
    auto env_ptr = std::make_unique<litert::Environment>(std::move(env));

    // 2. Create backend-specific options
    litert::Options litert_options = CreateNpuOptions();

    // 3. Call base initializer
    return TextEnhancer_Initialize_Base(options, std::move(litert_options), std::move(env_ptr));
}

/**
 * @brief Runs the inference (NPU Backend).
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