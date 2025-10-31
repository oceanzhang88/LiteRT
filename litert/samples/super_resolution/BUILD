load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//litert/build_common:special_rule.bzl",
    "gl_native_deps",
    "gles_deps",
    "gles_linkopts",
    "litert_android_linkopts",
)

package(
    default_visibility = ["//visibility:public"],
)

# --- Common File Groups ---

filegroup(
    name = "shader_files",
    srcs = [
        "shaders/passthrough_shader.vert",
        "shaders/super_res_compute.glsl",
        "shaders/crop_resize.comp",  # <-- Changed from .spv to .comp to match uploaded file
        "shaders/crop_resize.spv",   # <-- ADDED .spv file from your upload list
    ],
)

filegroup(
    name = "test_images",
    srcs = [
        "test_images/low_res_image.png",
        "test_images/low_res_image2.jpeg", # Added this based on your file list
        # --- ADDED other test images from your upload list ---
        "test_images/low_res_image3.png",
        "test_images/00000_1618x1328_0006 (1).png",
        "test_images/hand_art_sculpture_120875_1500x1500.jpg",
    ],
)

filegroup(
    name = "model_files",
    srcs = [
        "models/super_res-float.tflite",
        "models/real_x4v3-float.tflite",
        "models/real_x4v3_8650-float.tflite",
        "models/real_x4v3_8750-float.tflite",
        "models/real_x4v3-w8a8.tflite",
    ],
)

# --- Common C API Header ---
cc_library(
    name = "super_res_api",
    hdrs = ["super_res_api.h"],
)

# --- GPU Targets ---

cc_library(
    name = "super_res_lib_gpu_impl",
    srcs = ["backends/main_gpu.cc"],  # <-- Corrected path
    copts = ["-I."],
    linkopts = litert_android_linkopts() + gles_linkopts(),
    deps = [
        # ":image_processor", # --- REMOVED ---
        ":vulkan_image_processor", # --- ADDED ---
        ":image_utils",
        ":super_res_api",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
    ] + gles_deps() + gl_native_deps(),
)

cc_binary(
    name = "super_res_lib_gpu.so",
    linkshared = True,
    deps = [":super_res_lib_gpu_impl"],
)

cc_binary(
    name = "super_res_standalone_gpu",
    srcs = ["main_standalone_gpu.cc"],
    copts = ["-I."],
    data = [
        ":model_files",
        ":shader_files", # --- ADDED (for Vulkan preprocessor) ---
        ":super_res_lib_gpu.so",
        ":test_images",
        "//litert/c:litert_runtime_c_api_shared_lib",
        # --- KEPT: This is the correct GPU accelerator library ---
        "@litert_gpu//:jni/arm64-v8a/libLiteRtGpuAccelerator.so",
    ],
    linkopts = litert_android_linkopts(),
    deps = [
        ":image_utils",
        ":super_res_lib_gpu_impl",
    ],
)

# --- CPU Targets ---

cc_library(
    name = "super_res_lib_cpu_impl",
    srcs = ["backends/main_cpu.cc"], # This path was already correct
    copts = ["-I."],
    linkopts = litert_android_linkopts(),
    deps = [
        ":image_utils",
        ":super_res_api",
        ":vulkan_image_processor",  # This dependency is correct
        "//litert/cc:litert_api_with_dynamic_runtime",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
    ],
)

cc_binary(
    name = "super_res_lib_cpu.so",
    linkshared = True,
    deps = [":super_res_lib_cpu_impl"],
)

cc_binary(
    name = "super_res_standalone_cpu",
    srcs = ["main_standalone_cpu.cc"],
    copts = ["-I."],
    data = [
        ":model_files",
        ":shader_files", # --- ADDED (was missing, but main_standalone_cpu.cc needs it) ---
        ":super_res_lib_cpu.so",
        ":test_images",
        "//litert/c:litert_runtime_c_api_shared_lib",
    ],
    linkopts = litert_android_linkopts(),
    deps = [
        ":image_utils",
        ":super_res_lib_cpu_impl",
        # "//litert/tools:benchmark_litert_model",  # <-- This was commented out in your file, leaving it commented
    ],
)

# --- NPU Targets ---

cc_library(
    name = "super_res_lib_npu_impl",
    srcs = ["backends/main_npu.cc"],  # <-- Corrected path
    copts = ["-I."],
    deps = [
        ":image_utils",
        ":super_res_api",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
    ],
)

cc_binary(
    name = "super_res_lib_npu.so",
    linkshared = True,
    deps = [":super_res_lib_npu_impl"],
)

cc_binary(
    name = "super_res_standalone_npu",
    srcs = ["main_standalone_npu.cc"],
    copts = ["-I."],
    data = [
        ":model_files",
        ":super_res_lib_npu.so",
        ":test_images",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/vendors/qualcomm/dispatch:dispatch_api_so",
        "@qairt//:lib/aarch64-android/libQnnHtp.so",
        "@qairt//:lib/aarch64-android/libQnnHtpPrepare.so",
        "@qairt//:lib/aarch64-android/libQnnHtpV69Stub.so",
        "@qairt//:lib/aarch64-android/libQnnHtpV75Stub.so",
        "@qairt//:lib/aarch64-android/libQnnHtpV79Stub.so",
        "@qairt//:lib/aarch64-android/libQnnSystem.so",
        "@qairt//:lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so",
        "@qairt//:lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so",
        "@qairt//:lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so",
    ],
    deps = [
        ":image_utils",
        ":super_res_lib_npu_impl",
    ],
)

# --- Common Libraries ---

cc_library(
    name = "image_utils",
    srcs = ["utils/image_utils.cc"],  # <-- Corrected path
    hdrs = ["utils/image_utils.h"],  # <-- Corrected path
    deps = [
        "@stblib//:stb_image_hdrs",
        # # --- ADDED: AHB dependency for Android functions ---
        # "//litert/c:litert_c_api", 
    ],
)

# --- ADD THIS NEW TARGET ---
cc_library(
    name = "vulkan_image_processor",
    srcs = ["image_processing/vulkan_image_processor.cc"],
    hdrs = ["image_processing/vulkan_image_processor.h"],
    copts = ["-I."],
    # includes = ["image_processing/"], # <-- ADD THIS (This syntax is not standard, copts = ["-I."] is correct)
    linkopts = [
        "-lvulkan",  # Links libvulkan.so from the Android NDK
        "-llog",     # Links liblog.so for std::cout/cerr
    ],
    deps = [
        # Add dependencies on our new modular libraries
        "//litert/samples/super_resolution/image_processing/vulkan:vulkan_compute_pipeline",
        "//litert/samples/super_resolution/image_processing/vulkan:vulkan_context",
        "//litert/samples/super_resolution/image_processing/vulkan:vulkan_utils",
        
        # Keep existing dependencies
        "@vulkan_headers",
        # --- ADDED: AHB dependency for Android functions ---
        # "//litert/c:litert_c_api", 
    ],
)

# --- REMOVED OLD OPENGL PROCESSOR ---
# (You can delete this target if it's not used by anything else)
# cc_library(
#     name = "image_processor",
#     ...
# )