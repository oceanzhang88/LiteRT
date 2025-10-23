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
    ],
)

filegroup(
    name = "test_images",
    srcs = [
        "test_images/low_res_image.png",
    ],
)

filegroup(
    name = "model_files",
    srcs = [
        "models/super_res-float.tflite",
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
    srcs = ["main_gpu.cc"],
    copts = ["-I."],
    linkopts = litert_android_linkopts() + gles_linkopts(),
    deps = [
        ":image_processor",
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
        ":shader_files",
        ":super_res_lib_gpu.so",
        ":test_images",
        "//litert/c:litert_runtime_c_api_shared_lib",
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
    srcs = ["main_cpu.cc"],
    copts = ["-I."],
    linkopts = litert_android_linkopts(),
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
        ":super_res_lib_cpu.so",
        ":test_images",
        "//litert/c:litert_runtime_c_api_shared_lib",
    ],
    linkopts = litert_android_linkopts(),
    deps = [
        ":image_utils",
        ":super_res_lib_cpu_impl",
    ],
)

# --- NPU Targets ---

cc_library(
    name = "super_res_lib_npu_impl",
    srcs = ["main_npu.cc"],
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
    srcs = ["image_utils.cc"],
    hdrs = ["image_utils.h"],
    deps = [
        "@stblib//:stb_image_hdrs",
    ],
)

cc_library(
    name = "image_processor",
    srcs = ["image_processor.cc"],
    hdrs = ["image_processor.h"],
    linkopts = gles_linkopts(),
    deps = [
        ":image_utils",
        "@com_google_absl//absl/log:absl_check",
    ] + gles_deps(),
)