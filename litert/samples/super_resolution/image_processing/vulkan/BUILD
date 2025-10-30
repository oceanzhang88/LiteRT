# super_resolution/image_processing/vulkan/BUILD
#
# Defines the core, modular Vulkan components.

package(default_visibility = ["//visibility:public"])
# Or, for stricter visibility:
# package(default_visibility = ["//litert/samples/super_resolution/image_processing:__pkg__"])

licenses(["notice"])

# Library for low-level Vulkan helper functions
cc_library(
    name = "vulkan_utils",
    srcs = ["vulkan_utils.cc"],
    hdrs = ["vulkan_utils.h"],
    deps = [
        "@vulkan_headers",
    ],
)

# Library for the core Vulkan instance, device, and queue
cc_library(
    name = "vulkan_context",
    srcs = ["vulkan_context.cc"],
    hdrs = ["vulkan_context.h"],
    deps = [
        "@vulkan_headers",
    ],
)

# Library for compiling and managing the compute pipeline
cc_library(
    name = "vulkan_compute_pipeline",
    srcs = ["vulkan_compute_pipeline.cc"],
    hdrs = ["vulkan_compute_pipeline.h"],
    deps = [
        ":vulkan_context",
        ":vulkan_utils",
        "@vulkan_headers",
    ],
)