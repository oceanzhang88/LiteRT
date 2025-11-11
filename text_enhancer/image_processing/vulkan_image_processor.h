#pragma once

#include <vulkan/vulkan.h>

#include <memory>
#include <string>
#include <vector> // [ADDED]

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

    // [ADDED] Number of buffers for pipelining
    static const int kMaxFramesInFlight = 2;

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

    // --- [REMOVED] PreprocessImage (replaced by Submit/Sync) ---

#ifdef __ANDROID__
    // --- [REMOVED] PreprocessImage (AHB) (replaced by Submit/Sync) ---

    /**
     * @brief [ADDED] Submits preprocessing work for an AHardwareBuffer.
     *
     * Waits for the specified buffer_index's fence (from N-2 frames ago)
     * to ensure resources are free, then submits new work.
     * Does NOT wait for the new work to complete.
     *
     * @param in_buffer Pointer to the source AHardwareBuffer.
     * @param in_width Width of the source image (must match AHB).
     * @param in_height Height of the source image (must match AHB).
     * @param buffer_index The index (0 or 1) of the resource set to use.
     * @return true on successful submission, false on failure.
     */
    bool SubmitPreprocessImage(AHardwareBuffer* in_buffer, int in_width, int in_height,
                               int buffer_index);

    // --- [OMITTED] PreprocessImage_ZeroCopy, GetOutputAhb (unchanged) ---
    bool PreprocessImage_ZeroCopy(AHardwareBuffer* in_buffer, int in_width, int in_height);
    AHardwareBuffer* GetOutputAhb() { return output_ahb_; }
#endif  // __ANDROID__

    /**
     * @brief [ADDED] Submits preprocessing work for a CPU buffer.
     *
     * Waits for the specified buffer_index's fence (from N-2 frames ago)
     * to ensure resources are free, then submits new work.
     * Does NOT wait for the new work to complete.
     *
     * @param in_data Pointer to the source image data (unsigned char).
     * @param in_width Width of the source image.
     * @param in_height Height of the source image.
     * @param in_channels Channels of the source image (e.g., 4 for RGBA).
     * @param buffer_index The index (0 or 1) of the resource set to use.
     * @return true on successful submission, false on failure.
     */
    bool SubmitPreprocessImage(const unsigned char* in_data, int in_width, int in_height,
                               int in_channels, int buffer_index);

    /**
     * @brief [ADDED] Waits for preprocessing work and reads back the result.
     *
     * Waits for the fence associated with buffer_index to signal,
     * then copies the data from the readback buffer to out_data.
     *
     * @param out_data Pointer to the destination buffer (float* or int8_t*).
     * @param buffer_index The index (0 or 1) of the resource set to sync.
     * @return true on success, false on failure.
     */
    bool SyncPreprocess(void* out_data, int buffer_index);

    /**
     * @brief [MODIFIED] Public getter for last timings for a specific buffer.
     */
    TimingInfo GetLastTimings(int buffer_index) const {
        return last_timings_[buffer_index];
    }

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

    // --- [MODIFIED] Persistent Staging/Input Resources ---
    // Staging buffers are double-buffered
    std::vector<VkBuffer> staging_buffers_;
    std::vector<VkDeviceMemory> staging_buffers_memory_;

    // Input image is single-buffered (overwritten each frame)
    VkImage in_image_ = VK_NULL_HANDLE;
    VkDeviceMemory in_image_memory_ = VK_NULL_HANDLE;
    VkImageView in_image_view_ = VK_NULL_HANDLE;

    // --- [MODIFIED] Persistent Output/Readback Resources (Double-buffered) ---
    std::vector<VkBuffer> output_buffers_device_;
    std::vector<VkDeviceMemory> output_buffers_device_memory_;
    std::vector<VkBuffer> readback_buffers_;
    std::vector<VkDeviceMemory> readback_buffers_memory_;

    // --- [MODIFIED] Persistent Common Resources (Double-buffered) ---
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE; // Single pool
    std::vector<VkDescriptorSet> descriptor_sets_;
    std::vector<VkFence> fences_;
    std::vector<VkCommandBuffer> command_buffers_; // [ADDED]
    std::vector<TimingInfo> last_timings_;

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
    // [MODIFIED]
    bool createDescriptorSets();
};