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

# --- GPU Targets ---

cc_binary(
    name = "super_res_lib_gpu.so",
    srcs = ["main_gpu.cc"],
    copts = ["-I."],
    linkopts = litert_android_linkopts() + gles_linkopts(),
    linkshared = True,
    deps = [
        ":image_processor",
        ":image_utils",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",  # CORRECTED dependency name
    ] + gles_deps() + gl_native_deps(),
)

cc_binary(
    name = "super_res_loader_gpu",
    srcs = ["main_loader_gpu.cc"],
    data = [
        ":model_files",
        ":shader_files",
        ":super_res_lib_gpu.so",
        ":test_images",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "@litert_gpu//:jni/arm64-v8a/libLiteRtGpuAccelerator.so",
    ],
    linkopts = ["-ldl"] + litert_android_linkopts(),
)

# --- CPU Targets ---

cc_binary(
    name = "super_res_lib_cpu.so",
    srcs = ["main_cpu.cc"],
    copts = ["-I."],
    linkopts = litert_android_linkopts() + gles_linkopts(),
    linkshared = True,
    deps = [
        ":image_processor",
        ":image_utils",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",  # CORRECTED dependency name
    ] + gles_deps() + gl_native_deps(),
)

cc_binary(
    name = "super_res_loader_cpu",
    srcs = ["main_loader_cpu.cc"],
    data = [
        ":model_files",
        ":shader_files",
        ":super_res_lib_cpu.so",
        ":test_images",
        "//litert/c:litert_runtime_c_api_shared_lib",
    ],
    linkopts = ["-ldl"] + litert_android_linkopts(),
)

# --- NPU Targets ---

cc_binary(
    name = "super_res_lib_npu.so",
    srcs = ["main_npu.cc"],
    copts = ["-I."],
    linkopts = gles_linkopts(),
    linkshared = True,
    deps = [
        ":image_processor",
        ":image_utils",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",  # CORRECTED dependency name
    ] + gles_deps() + gl_native_deps(),
)

cc_binary(
    name = "super_res_loader_npu",
    srcs = ["main_loader_npu.cc"],
    data = [
        ":model_files",
        ":shader_files",
        ":super_res_lib_npu.so",
        ":test_images",
        "//litert/c:litert_runtime_c_api_shared_lib",
        "//litert/vendors/qualcomm/dispatch:dispatch_api_so",
        "@qairt//:lib/aarch64-android/libQnnHtp.so",
        "@qairt//:lib/aarch64-android/libQnnHtpPrepare.so",
        "@qairt//:lib/aarch64-android/libQnnHtpV75Stub.so",
        "@qairt//:lib/aarch64-android/libQnnHtpV79Stub.so",
        "@qairt//:lib/aarch64-android/libQnnSystem.so",
        "@qairt//:lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so",
        "@qairt//:lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so",
    ],
    linkopts = ["-ldl"] + gles_linkopts(),
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
