load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_import.bzl", "cc_import")

load(
    "//litert/build_common:special_rule.bzl",
    "gl_native_deps",
    "gles_deps",
    # "gles_linkopts",
    "litert_android_linkopts",
)

package(
    default_visibility = ["//visibility:public"],
)

# --- Common File Groups ---
# ... (filegroups unchanged) ...
filegroup(
    name = "shader_files",
    srcs = [
        "shaders/passthrough_shader.vert",
        "shaders/super_res_compute.glsl",
        "shaders/crop_resize_float.comp",
        "shaders/crop_resize_float.spv",
        "shaders/crop_resize_uint8.comp",
        "shaders/crop_resize_uint8.spv",
    ],
)

filegroup(
    name = "test_images",
    srcs = [
        "test_images/low_res_image.png",
        "test_images/low_res_image2.jpeg",
        "test_images/low_res_image3.png",
        "test_images/00000_1618x1328_0006 (1).png",
        "test_images/hand_art_sculpture_120875_1500x1500.jpg",
    ],
)

filegroup(
    name = "model_files",
    srcs = [
        "models/super_res-float_npu.tflite",
        "models/super_res-float_npu_vst.tflite",
        "models/super_res-float_gpu.tflite",
        "models/real_x4v3-float.tflite",
        "models/real_x4v3_8650-float.tflite",
        "models/real_x4v3_8750-float.tflite",
        "models/real_x4v3-w8a8.tflite",
        "models/super_res-w8a8_SM8450.tflite",
        # "models/selfie_multiclass_256x256_SM8750.tflite",
    ],
)

# --- C API Header ---
cc_library(
    name = "text_enhancer_api",
    hdrs = ["text_enhancer_api.h"],
)

# --- Base Backend Library ---
# ... (text_enhancer_session_base unchanged) ...
cc_library(
    name = "text_enhancer_session_base",
    srcs = ["backends/base/text_enhancer_session_base.cc"],
    hdrs = ["backends/base/text_enhancer_session_base.h"],
    copts = ["-I."],
    deps = [
        ":image_utils",
        ":text_enhancer_api",
        ":vulkan_image_processor",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
        "@com_google_absl//absl/log",
    ],
)

# --- CPU Targets ---
# ... (text_enhancer_lib_cpu_impl unchanged) ...
cc_library(
    name = "text_enhancer_lib_cpu_impl",
    srcs = ["backends/text_enhancer_cpu.cc"], 
    copts = ["-I."],
    linkopts = litert_android_linkopts(),
    data = [],
    deps = [
        ":text_enhancer_session_base",
    ],
)

cc_binary(
    name = "text_enhancer_lib_cpu.so",
    linkshared = True,
    data = [],
    deps = [":text_enhancer_lib_cpu_impl"],
)

# --- GPU Targets ---
# ... (text_enhancer_lib_gpu_impl unchanged) ...
cc_library(
    name = "text_enhancer_lib_gpu_impl",
    srcs = ["backends/text_enhancer_gpu.cc"], 
    copts = ["-I."],
    linkopts = litert_android_linkopts(),  # + gles_linkopts(),
    data = [],
    deps = [
        ":text_enhancer_session_base",
    ] + gles_deps() + gl_native_deps(),
)

cc_binary(
    name = "text_enhancer_lib_gpu.so",
    linkshared = True,
    data = [
        "//litert/c:litert_runtime_c_api_shared_lib",  # buildcleaner: keep
        "@litert_gpu//:jni/arm64-v8a/libLiteRtOpenClAccelerator.so",  # copybara:comment
    ],
    deps = [
        ":text_enhancer_lib_gpu_impl",
    ],
)

# --- NPU Targets ---
# ... (text_enhancer_lib_npu_impl unchanged) ...
cc_library(
    name = "text_enhancer_lib_npu_impl",
    srcs = ["backends/text_enhancer_npu.cc"], 
    copts = ["-I."],
    linkopts = litert_android_linkopts(),
    data = [],
    deps = [
        ":text_enhancer_session_base",
    ],
)

cc_binary(
    name = "text_enhancer_lib_npu.so",
    linkshared = True,
    data = [
        "//litert/vendors/qualcomm/dispatch:dispatch_api_so",
        "@qairt//:lib/aarch64-android/libQnnHtp.so",
        "@qairt//:lib/aarch64-android/libQnnHtpPrepare.so",
        "@qairt//:lib/aarch64-android/libQnnSystem.so",
        "@qairt//:lib/aarch64-android/libQnnHtpV69Stub.so",
        "@qairt//:lib/aarch64-android/libQnnHtpV75Stub.so",
        "@qairt//:lib/aarch64-android/libQnnHtpV79Stub.so",
        "@qairt//:lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so",
        "@qairt//:lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so",
        "@qairt//:lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so",
    ],
    deps = [
        ":text_enhancer_lib_npu_impl",
    ],
)

# --- Dummy Targets ---
# ... (text_enhancer_lib_dummy_impl unchanged) ...
cc_library(
    name = "text_enhancer_lib_dummy_impl",
    srcs = ["backends/text_enhancer_dummy.cc"], 
    copts = ["-I."],
    deps = [
        ":text_enhancer_api",
        "@com_google_absl//absl/log",  
    ],
)

cc_binary(
    name = "text_enhancer_lib_dummy.so",
    linkshared = True,
    deps = [":text_enhancer_lib_dummy_impl"],
)


# --- Common Libraries (Internal paths are fine) ---
# ... (image_utils, vulkan_image_processor unchanged) ...
cc_library(
    name = "image_utils",
    srcs = ["utils/image_utils.cc"],
    hdrs = ["utils/image_utils.h"],
    deps = [
        "@stblib//:stb_image_hdrs",
        # --- Add absl/log here too if image_utils uses it ---
        # "@com_google_absl//absl/log", 
    ],
)

cc_library(
    name = "vulkan_image_processor",
    srcs = ["image_processing/vulkan_image_processor.cc"],
    hdrs = ["image_processing/vulkan_image_processor.h"],
    copts = ["-I."],
    linkopts = [
        "-lvulkan",
        "-llog", # This is for Android log, can be kept
    ],
    deps = [
        "//text_enhancer/image_processing/vulkan:vulkan_compute_pipeline",
        "//text_enhancer/image_processing/vulkan:vulkan_context",
        "//text_enhancer/image_processing/vulkan:vulkan_utils",
        "@vulkan_headers",
        "@com_google_absl//absl/log",  
    ],
)


# --- NEW: Common Standalone Library ---
cc_library(
    name = "main_standalone_common",
    hdrs = ["main_standalone_common.h"],
    deps = [
        ":text_enhancer_api",
        ":image_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/log",
    ],
)

# --- MODIFIED: Main Binaries (Standalone Examples) ---

cc_binary(
    name = "text_enhancer_standalone_cpu",
    srcs = ["main_standalone_cpu.cc"], # MODIFIED
    copts = ["-I."],
    data = [
        ":model_files",
        ":shader_files",
        ":text_enhancer_lib_cpu.so",
        ":test_images",
    ],
    linkopts = litert_android_linkopts() + ["-ldl"],
    deps = [
        ":main_standalone_common", # MODIFIED
    ],
)

cc_binary(
    name = "text_enhancer_standalone_gpu",
    srcs = ["main_standalone_gpu.cc"], # MODIFIED
    copts = ["-I."],
    data = [
        ":model_files",
        ":shader_files",
        ":text_enhancer_lib_gpu.so",
        ":test_images",
    ],
    linkopts = litert_android_linkopts() + [
        "-ldl",
        "-Wl,--export-dynamic",  
        ],
    deps = [
        ":main_standalone_common", # MODIFIED
    ],
)

cc_binary(
    name = "text_enhancer_standalone_npu",
    srcs = ["main_standalone_npu.cc"], # MODIFIED
    copts = ["-I."],
    data = [
        ":model_files",
        ":text_enhancer_lib_npu.so",
        ":test_images",
    ],
    linkopts = litert_android_linkopts() + ["-ldl"],
    deps = [
        ":main_standalone_common", # MODIFIED
    ],
)

cc_binary(
    name = "text_enhancer_standalone_dummy",
    srcs = ["main_standalone_dummy.cc"], # MODIFIED
    copts = ["-I."],
    data = [
        ":text_enhancer_lib_dummy.so",
    ],
    linkopts = litert_android_linkopts() + ["-ldl"], 
    deps = [
        ":main_standalone_common", # MODIFIED
    ],
)