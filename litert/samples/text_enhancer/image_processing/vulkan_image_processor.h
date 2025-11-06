#ifndef THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_SUPER_RES_VULKAN_IMAGE_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_SUPER_RES_VULKAN_IMAGE_PROCESSOR_H_

#include <string>
#include <vector>
#include <memory>
#include <vulkan/vulkan.h>
#include "vulkan/vulkan_context.h"
#include "vulkan/vulkan_compute_pipeline.h"

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#include <vulkan/vulkan_android.h> // For PFN types
#endif


// Note: CropResizePushConstants is now defined in vulkan_compute_pipeline.h

class VulkanImageProcessor {
   public:
    VulkanImageProcessor();
    ~VulkanImageProcessor();

    /**
     * @brief Initializes the Vulkan context and compute pipeline.
     * @param shader_spirv_path Path to the compiled SPIR-V shader file.
     * @param out_width Target width (e.g., 256).
     * @param out_height Target height (e.g., 256).
     * @return true on success, false on failure.
     */
    bool Initialize(const std::string& shader_spirv_path,
                    int out_width, int out_height);

    /**
     * @brief Shuts down all Vulkan resources.
     */
    void Shutdown();

    /**
     * @brief Performs center-crop and resize using the Vulkan compute shader.
     *
     * @param in_data Pointer to the source image data (unsigned char).
     * @param in_width Width of the source image.
     * @param in_height Height of the source image.
     * @param in_channels Channels of the source image (e.g., 4 for RGBA).
     * @param out_data Pointer to the destination buffer (float).
     * @return true on success, false on failure.
     */
    bool PreprocessImage(const unsigned char* in_data,
                         int in_width, int in_height, int in_channels,
                         float* out_data);

    #ifdef __ANDROID__
    /**
     * @brief Performs center-crop and resize using an AHardwareBuffer as input.
     * (NOTE: This version performs a CPU readback.)
     *
     * @param in_buffer Pointer to the source AHardwareBuffer.
     * @param in_width Width of the source image (must match AHB).
     * @param in_height Height of the source image (must match AHB).
     * @param out_data Pointer to the destination buffer (float).
     * @return true on success, false on failure.
     */
    bool PreprocessImage(AHardwareBuffer* in_buffer,
                         int in_width, int in_height,
                         float* out_data);

    /**
     * @brief Performs center-crop and resize (Zero-Copy path).
     * This populates the internal output AHardwareBuffer.
     *
     * @param in_buffer Pointer to the source AHardwareBuffer.
     * @param in_width Width of the source image (must match AHB).
     * @param in_height Height of the source image (must match AHB).
     * @return true on success, false on failure.
     */
    bool PreprocessImage_ZeroCopy(AHardwareBuffer* in_buffer,
                                  int in_width, int in_height);

    /**
     * @brief Gets the handle to the persistent output AHardwareBuffer.
     * This buffer is the destination for the Vulkan compute shader.
     * @return AHardwareBuffer handle, or nullptr if not initialized.
     */
    AHardwareBuffer* GetOutputAhb() { return output_ahb_; }

    #endif // __ANDROID__


   private:
    // --- Core Vulkan Modules ---
    std::unique_ptr<VulkanContext> context_;
    std::unique_ptr<VulkanComputePipeline> compute_pipeline_;

    // --- Persistent Resources ---
    int out_width_ = 0;
    int out_height_ = 0;
    VkDeviceSize out_size_bytes_ = 0;

    // Readback (GPU -> CPU)
    VkBuffer readback_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory readback_buffer_memory_ = VK_NULL_HANDLE;

    // Output Image (GPU-local)
    VkImage out_image_ = VK_NULL_HANDLE;
    VkDeviceMemory out_image_memory_ = VK_NULL_HANDLE;
    VkImageView out_image_view_ = VK_NULL_HANDLE;

    // Sampler (re-used for all inputs)
    VkSampler sampler_ = VK_NULL_HANDLE;
    
    // Descriptor Pool & Set
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    
    // Synchronization
    VkFence fence_ = VK_NULL_HANDLE;

    // --- AHB Zero-Copy Output ---
    #ifdef __ANDROID__
    AHardwareBuffer* output_ahb_ = nullptr;
    #endif

    #ifdef __ANDROID__
    // --- AHB-related Function Pointers ---
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAndroidHardwareBufferPropertiesANDROID_ = nullptr;
    PFN_vkGetMemoryAndroidHardwareBufferANDROID vkGetMemoryAndroidHardwareBufferANDROID_ = nullptr;
    #endif

    // --- Private Helper Functions ---
    bool createPersistentResources();
    void destroyPersistentResources();
    bool createDescriptorPool();
    bool createDescriptorSet();
};

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_SUPER_RES_VULKAN_IMAGE_PROCESSOR_H_