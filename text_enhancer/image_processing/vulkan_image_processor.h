#pragma once

#include <vulkan/vulkan.h>

#include <memory>
#include <string>

#include "vulkan/vulkan_compute_pipeline.h"
#include "vulkan/vulkan_context.h"

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#include <vulkan/vulkan_android.h>  // For PFN types
#endif

// Note: CropResizePushConstants is defined in vulkan_compute_pipeline.h

class VulkanImageProcessor {
   public:
    // --- Public struct for detailed timings ---
    struct TimingInfo {
        // --- CPU Timings ---
        double staging_copy_ms = 0.0;    // CPU memcpy to staging
        double readback_copy_ms = 0.0;   // CPU memcpy from readback
        
        // --- CPU Wait Time ---
        double gpu_submit_wait_ms = 0.0; // Total vkWaitForFences
        
        // --- GPU-Only Timings (from vkCmdWriteTimestamp) ---
        double gpu_shader_ms = 0.0;      // Time for vkCmdDispatch (the compute shader).
        double gpu_readback_ms = 0.0;    // Time for vkCmdCopyBuffer (device to host buffer).
    };

    VulkanImageProcessor();
    ~VulkanImageProcessor();

    /**
     * @brief Initializes the Vulkan context and compute pipeline.
     *
     * @param shader_spirv_path Path to the compiled SPIR-V shader file.
     * @param max_in_width Maximum expected input width.
     * @param max_in_height Maximum expected input height.
     * @param max_in_channels Maximum expected input channels (e.g., 4).
     * @param out_width Target width (e.g., 256).
     * @param out_height Target height (e.g., 256).
     * @param is_output_int8 True if the output is int8, false if float.
     * @return true on success, false on failure.
     */
    bool Initialize(const std::string& shader_spirv_path, int max_in_width, int max_in_height,
                    int max_in_channels, int out_width, int out_height, bool is_output_int8);

    /**
     * @brief Shuts down all Vulkan resources.
     */
    void Shutdown();

    /**
     * @brief Performs center-crop and resize using the Vulkan compute shader.
     * (This path is now optimized to re-use persistent resources)
     *
     * @param in_data Pointer to the source image data (unsigned char).
     * @param in_width Width of the source image.
     * @param in_height Height of the source image.
     * @param in_channels Channels of the source image (e.g., 4 for RGBA).
     * @param out_data Pointer to the destination buffer (float* or int8_t*).
     * @return true on success, false on failure.
     */
    bool PreprocessImage(const unsigned char* in_data, int in_width, int in_height, int in_channels,
                         void* out_data);

#ifdef __ANDROID__
    /**
     * @brief Performs center-crop and resize using an AHardwareBuffer as input.
     * (This path is now optimized to cache the imported AHB)
     *
     * @param in_buffer Pointer to the source AHardwareBuffer.
     * @param in_width Width of the source image (must match AHB).
     * @param in_height Height of the source image (must match AHB).
     * @param out_data Pointer to the destination buffer (float* or int8_t*).
     * @return true on success, false on failure.
     */
    bool PreprocessImage(AHardwareBuffer* in_buffer, int in_width, int in_height, void* out_data);

    /**
     * @brief Performs center-crop and resize (Zero-Copy path).
     * This populates the internal output AHardwareBuffer.
     *
     * @param in_buffer Pointer to the source AHardwareBuffer.
     * @param in_width Width of the source image (must match AHB).
     * @param in_height Height of the source image (must match AHB).
     * @return true on success, false on failure.
     */
    bool PreprocessImage_ZeroCopy(AHardwareBuffer* in_buffer, int in_width, int in_height);

    /**
     * @brief Gets the handle to the persistent output AHardwareBuffer.
     * This buffer is the destination for the Vulkan compute shader.
     * @return AHardwareBuffer handle, or nullptr if not initialized.
     */
    AHardwareBuffer* GetOutputAhb() { return output_ahb_; }

#endif  // __ANDROID__

    // --- Public getter for last timings ---
    TimingInfo GetLastTimings() const { return last_timings_; }

   private:
    // --- Core Vulkan Modules ---
    std::unique_ptr<VulkanContext> context_;
    std::unique_ptr<VulkanComputePipeline> compute_pipeline_;

    // --- Output Image Properties ---
    int out_width_ = 0;
    int out_height_ = 0;
    VkDeviceSize out_size_bytes_ = 0;
    bool is_output_int8_ = false;

    // --- Persistent Input Image Properties ---
    int max_in_width_ = 0;
    int max_in_height_ = 0;
    VkDeviceSize in_staging_size_bytes_ = 0;
    VkFormat in_image_format_ = VK_FORMAT_UNDEFINED;

    // --- Persistent Staging/Input Resources ---
    VkBuffer staging_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory staging_buffer_memory_ = VK_NULL_HANDLE;

    VkImage in_image_ = VK_NULL_HANDLE;
    VkDeviceMemory in_image_memory_ = VK_NULL_HANDLE;
    VkImageView in_image_view_ = VK_NULL_HANDLE;

    // --- Persistent Output/Readback Resources ---
    VkBuffer output_buffer_device_ = VK_NULL_HANDLE;
    VkDeviceMemory output_buffer_device_memory_ = VK_NULL_HANDLE;
    VkBuffer readback_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory readback_buffer_memory_ = VK_NULL_HANDLE;

    // --- Persistent Common Resources ---
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    TimingInfo last_timings_ = {};

    // --- Persistent AHB Input Resources (Cache) ---
    AHardwareBuffer* last_in_ahb_ = nullptr;
    VkImage ahb_in_image_ = VK_NULL_HANDLE;
    VkDeviceMemory ahb_in_image_memory_ = VK_NULL_HANDLE;
    VkImageView ahb_in_image_view_ = VK_NULL_HANDLE;


// --- AHB Zero-Copy Output ---
#ifdef __ANDROID__
    AHardwareBuffer* output_ahb_ = nullptr;
#endif

#ifdef __ANDROID__
    // --- AHB-related Function Pointers ---
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAndroidHardwareBufferPropertiesANDROID_ =
        nullptr;
    PFN_vkGetMemoryAndroidHardwareBufferANDROID vkGetMemoryAndroidHardwareBufferANDROID_ = nullptr;
#endif

    // --- Private Helper Functions ---
    bool createPersistentResources();
    void destroyPersistentResources();
    void destroyAhbInputResources();
    bool createDescriptorPool();
    bool createDescriptorSet();
};