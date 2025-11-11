#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vulkan/vulkan_utils.h"  // Our new utils header
#include "vulkan_image_processor.h"

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
// Note: vulkan_android.h is included by vulkan_image_processor.h
#endif

// --- Constructor, Destructor, Initialize, Shutdown ---
// [OMITTED FOR BREVITY - MODIFIED INITIALIZE/SHUTDOWN BELOW]
VulkanImageProcessor::VulkanImageProcessor() {
    // All members initialized in header
}
VulkanImageProcessor::~VulkanImageProcessor() { Shutdown(); }
bool VulkanImageProcessor::Initialize(const std::string& shader_spirv_path, int max_in_width,
                                      int max_in_height, int max_in_channels, int out_width,
                                      int out_height, bool is_output_int8) {
    // --- [MODIFIED] Resize vectors ---
    staging_buffers_.resize(kMaxFramesInFlight);
    staging_buffers_memory_.resize(kMaxFramesInFlight);
    output_buffers_device_.resize(kMaxFramesInFlight);
    output_buffers_device_memory_.resize(kMaxFramesInFlight);
    readback_buffers_.resize(kMaxFramesInFlight);
    readback_buffers_memory_.resize(kMaxFramesInFlight);
    descriptor_sets_.resize(kMaxFramesInFlight);
    fences_.resize(kMaxFramesInFlight);
    command_buffers_.resize(kMaxFramesInFlight);
    last_timings_.resize(kMaxFramesInFlight);
    // --- END MODIFIED ---

    try {
        out_width_ = out_width;
        out_height_ = out_height;
        is_output_int8_ = is_output_int8;
        max_in_width_ = max_in_width;
        max_in_height_ = max_in_height;
        if (is_output_int8_) {
            out_size_bytes_ =
                static_cast<VkDeviceSize>(out_width) * out_height * 3 * sizeof(int8_t);
            std::cout << "Vulkan processor configured for INT8 output." << std::endl;
        } else {
            out_size_bytes_ = static_cast<VkDeviceSize>(out_width) * out_height * 3 * sizeof(float);
            std::cout << "Vulkan processor configured for FLOAT32 output." << std::endl;
        }
        std::cout << "Output buffer size: " << out_size_bytes_ << " bytes." << std::endl;
        in_staging_size_bytes_ =
            static_cast<VkDeviceSize>(max_in_width) * max_in_height * max_in_channels;
        in_image_format_ = VK_FORMAT_R8G8B8A8_UNORM;  // Hardcode RGBA8 input
        std::cout << "Persistent input staging buffer size: " << in_staging_size_bytes_
                  << " bytes." << std::endl;
        context_ = std::make_unique<VulkanContext>();
        if (!context_->Initialize()) {
            throw std::runtime_error("Failed to initialize VulkanContext.");
        }
#ifdef __ANDROID__
        vkGetAndroidHardwareBufferPropertiesANDROID_ =
            (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)vkGetDeviceProcAddr(
                context_->GetDevice(), "vkGetAndroidHardwareBufferPropertiesANDROID");
        if (!vkGetAndroidHardwareBufferPropertiesANDROID_) {
            throw std::runtime_error(
                "Failed to get vkGetAndroidHardwareBufferPropertiesANDROID proc addr.");
        }
        vkGetMemoryAndroidHardwareBufferANDROID_ =
            (PFN_vkGetMemoryAndroidHardwareBufferANDROID)vkGetDeviceProcAddr(
                context_->GetDevice(), "vkGetMemoryAndroidHardwareBufferANDROID");
        if (!vkGetMemoryAndroidHardwareBufferANDROID_) {
            throw std::runtime_error(
                "Failed to get vkGetMemoryAndroidHardwareBufferANDROID proc addr.");
        }
        std::cout << "Loaded AHB extension functions." << std::endl;
#endif
        compute_pipeline_ = std::make_unique<VulkanComputePipeline>();
        if (!compute_pipeline_->Initialize(context_.get(), shader_spirv_path)) {
            throw std::runtime_error("Failed to initialize VulkanComputePipeline.");
        }
        if (!createPersistentResources()) {
            throw std::runtime_error("Failed to create persistent resources.");
        }
    } catch (const std::exception& e) {
        std::cerr << "VulkanImageProcessor Initialization Error: " << e.what() << std::endl;
        Shutdown();
        return false;
    }
    std::cout << "VulkanImageProcessor initialized successfully." << std::endl;
    return true;
}
void VulkanImageProcessor::Shutdown() {
    if (context_ && context_->GetDevice()) {
        vkDeviceWaitIdle(context_->GetDevice());
    }
    destroyPersistentResources();
    if (compute_pipeline_) {
        compute_pipeline_->Shutdown();
        compute_pipeline_.reset();
    }
    if (context_) {
        context_->Shutdown();
        context_.reset();
    }
}


// --- [REMOVED] PreprocessImage (CPU) ---
// --- [REMOVED] PreprocessImage (AHB) ---


// --- [ADDED] SubmitPreprocessImage (CPU Path) ---
bool VulkanImageProcessor::SubmitPreprocessImage(const unsigned char* in_data, int in_width,
                                                 int in_height, int in_channels,
                                                 int buffer_index) {
    if (!context_ || !compute_pipeline_) {
        std::cerr << "Processor not initialized." << std::endl;
        return false;
    }
    VkDevice device = context_->GetDevice();
    if (in_channels != 4) {
        std::cerr << "Vulkan processor currently only supports 4-channel RGBA images." << std::endl;
        return false;
    }
    const VkDeviceSize in_size_bytes =
        static_cast<VkDeviceSize>(in_width) * in_height * in_channels;
    if (in_size_bytes > in_staging_size_bytes_) {
        std::cerr << "Input image size (" << in_size_bytes
                  << ") exceeds persistent buffer size (" << in_staging_size_bytes_
                  << "). Re-initialize with larger max dims." << std::endl;
        return false;
    }

    // --- Reset timings for this frame ---
    last_timings_[buffer_index] = {};
    std::chrono::high_resolution_clock::time_point start_time, end_time, wait_start_time;
    
    VkQueryPool query_pool = context_->GetQueryPool();
    float timestamp_period_ns = context_->GetTimestampPeriod();
    bool can_query_timestamps = (query_pool != VK_NULL_HANDLE && timestamp_period_ns > 0.0f);

    try {
        // --- 1. Wait for this resource set to be free ---
        wait_start_time = std::chrono::high_resolution_clock::now();
        vkWaitForFences(device, 1, &fences_[buffer_index], VK_TRUE, UINT64_MAX);
        end_time = std::chrono::high_resolution_clock::now();
        last_timings_[buffer_index].gpu_submit_wait_ms =
            std::chrono::duration<double, std::milli>(end_time - wait_start_time).count();
        
        vkResetFences(device, 1, &fences_[buffer_index]);

        // --- 2. Copy Data to Staging Buffer ---
        start_time = std::chrono::high_resolution_clock::now();
        void* mapped_data = VulkanUtils::MapBufferMemory(
            device, staging_buffers_memory_[buffer_index], in_size_bytes);
        memcpy(mapped_data, in_data, in_size_bytes);
        VulkanUtils::UnmapBufferMemory(device, staging_buffers_memory_[buffer_index]);
        end_time = std::chrono::high_resolution_clock::now();
        last_timings_[buffer_index].staging_copy_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();

        // --- 3. Record & Submit Commands ---
        VkCommandBuffer cmd = command_buffers_[buffer_index];
        vkResetCommandBuffer(cmd, 0); // Reset for re-recording

        VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin_info);

        uint32_t query_idx_base = buffer_index * 4; // Use offset in query pool
        if (can_query_timestamps) {
            vkCmdResetQueryPool(cmd, query_pool, query_idx_base, 4);
        }

        // a. Copy staging buffer to input image
        VulkanUtils::TransitionImageLayout(cmd, in_image_, in_image_format_,
                                           VK_IMAGE_LAYOUT_GENERAL,
                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VulkanUtils::CopyBufferToImage(cmd, staging_buffers_[buffer_index], in_image_, in_width,
                                       in_height);

        // b. Transition input image for shader read
        VulkanUtils::TransitionImageLayout(cmd, in_image_, in_image_format_,
                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                           VK_IMAGE_LAYOUT_GENERAL);

        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query_idx_base + 0);
        }

        // c. Bind pipeline and descriptors
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_->GetPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compute_pipeline_->GetPipelineLayout(), 0, 1,
                                &descriptor_sets_[buffer_index], 0, nullptr);

        // d. Set push constants
        CropResizePushConstants constants = {};
        constants.in_dims[0] = in_width;
        constants.in_dims[1] = in_height;
        constants.crop_dims[0] = 512;
        constants.crop_dims[1] = 512;
        constants.out_dims[0] = out_width_;
        constants.out_dims[1] = out_height_;
        vkCmdPushConstants(cmd, compute_pipeline_->GetPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(CropResizePushConstants), &constants);

        // e. Dispatch compute job
        vkCmdDispatch(cmd, (out_width_ + 7) / 8, (out_height_ + 7) / 8, 1);

        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query_idx_base + 1);
        }

        // f. Barrier: Shader writes -> Transfer read
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                             nullptr);

        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, query_pool, query_idx_base + 2);
        }

        // g. Copy device-local buffer to host-visible buffer
        VkBufferCopy copy_region = {};
        copy_region.size = out_size_bytes_;
        vkCmdCopyBuffer(cmd, output_buffers_device_[buffer_index],
                        readback_buffers_[buffer_index], 1, &copy_region);

        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, query_pool, query_idx_base + 3);
        }
        
        // h. Barrier: Transfer write -> Host read (This is for Sync)
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                             1, &barrier, 0, nullptr, 0, nullptr);

        // i. End command buffer
        vkEndCommandBuffer(cmd);

        // j. Submit commands
        VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;

        if (vkQueueSubmit(context_->GetComputeQueue(), 1, &submit_info, fences_[buffer_index]) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to submit command buffer!");
        }

        // --- DO NOT WAIT HERE ---

    } catch (const std::exception& e) {
        std::cerr << "[Debug SubmitPreprocessImage] EXCEPTION CAUGHT: " << e.what() << std::endl;
        return false;
    }

    return true;
}

#ifdef __ANDROID__
// --- [ADDED] SubmitPreprocessImage (AHardwareBuffer Path) ---
bool VulkanImageProcessor::SubmitPreprocessImage(AHardwareBuffer* in_buffer, int in_width,
                                                 int in_height, int buffer_index) {
    if (!context_ || !compute_pipeline_ || !vkGetAndroidHardwareBufferPropertiesANDROID_ ||
        !vkGetMemoryAndroidHardwareBufferANDROID_) {
        std::cerr << "Processor not initialized or AHB extensions not loaded." << std::endl;
        return false;
    }
    VkDevice device = context_->GetDevice();

    // --- Reset timings for this frame ---
    last_timings_[buffer_index] = {};
    std::chrono::high_resolution_clock::time_point start_time, end_time, wait_start_time;
    
    VkQueryPool query_pool = context_->GetQueryPool();
    float timestamp_period_ns = context_->GetTimestampPeriod();
    bool can_query_timestamps = (query_pool != VK_NULL_HANDLE && timestamp_period_ns > 0.0f);
    
    VkFormat in_image_format = VK_FORMAT_UNDEFINED;

    try {
        // --- 1. Wait for this resource set to be free ---
        wait_start_time = std::chrono::high_resolution_clock::now();
        vkWaitForFences(device, 1, &fences_[buffer_index], VK_TRUE, UINT64_MAX);
        end_time = std::chrono::high_resolution_clock::now();
        last_timings_[buffer_index].gpu_submit_wait_ms =
            std::chrono::duration<double, std::milli>(end_time - wait_start_time).count();
        
        vkResetFences(device, 1, &fences_[buffer_index]);

        // --- 2. Import AHB (if needed) ---
        start_time = std::chrono::high_resolution_clock::now();
        if (in_buffer != last_in_ahb_ || ahb_in_image_ == VK_NULL_HANDLE) {
            std::cout << "[Debug PreprocessImage-AHB] New AHB handle detected. Caching..." << std::endl;
            vkDeviceWaitIdle(device); // Wait for all ops to finish before destroying
            destroyAhbInputResources();

            if (!VulkanUtils::ImportAhbToImage(
                    device, context_->GetPhysicalDevice(), in_buffer,
                    vkGetAndroidHardwareBufferPropertiesANDROID_,
                    vkGetMemoryAndroidHardwareBufferANDROID_,
                    VK_IMAGE_USAGE_STORAGE_BIT,
                    ahb_in_image_, ahb_in_image_memory_, ahb_in_image_view_, in_image_format)) {
                throw std::runtime_error("Failed to import AHardwareBuffer to VkImage.");
            }

            // Update *both* descriptor sets to point to the new AHB image
            // This is simpler than tracking which descriptor set used the AHB last.
            for (int i = 0; i < kMaxFramesInFlight; ++i) {
                VkDescriptorImageInfo input_image_info = {};
                input_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                input_image_info.imageView = ahb_in_image_view_;
                input_image_info.sampler = VK_NULL_HANDLE;

                VkWriteDescriptorSet write_input = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write_input.dstSet = descriptor_sets_[i]; // Update set [i]
                write_input.dstBinding = 0; 
                write_input.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_input.descriptorCount = 1;
                write_input.pImageInfo = &input_image_info;

                vkUpdateDescriptorSets(device, 1, &write_input, 0, nullptr);
            }
            
            // Transition layout
            VkCommandBuffer cmd = context_->BeginOneTimeCommands();
            VulkanUtils::TransitionImageLayout(cmd, ahb_in_image_, in_image_format,
                                            VK_IMAGE_LAYOUT_UNDEFINED,
                                            VK_IMAGE_LAYOUT_GENERAL);
            context_->EndAndSubmitCommands(cmd); 

            last_in_ahb_ = in_buffer;
            std::cout << "[Debug PreprocessImage-AHB] Caching complete." << std::endl;
        }
        end_time = std::chrono::high_resolution_clock::now();
        last_timings_[buffer_index].staging_copy_ms = 
            std::chrono::duration<double, std::milli>(end_time - start_time).count();


        // --- 3. Record & Submit Commands ---
        VkCommandBuffer cmd = command_buffers_[buffer_index];
        vkResetCommandBuffer(cmd, 0); 

        VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin_info);

        uint32_t query_idx_base = buffer_index * 4; 
        if (can_query_timestamps) {
            vkCmdResetQueryPool(cmd, query_pool, query_idx_base, 4);
        }

        // a. Transition (already in GENERAL)

        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query_idx_base + 0);
        }

        // b. Bind pipeline and descriptors
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_->GetPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compute_pipeline_->GetPipelineLayout(), 0, 1,
                                &descriptor_sets_[buffer_index], 0, nullptr);

        // c. Set push constants
        CropResizePushConstants constants = {};
        constants.in_dims[0] = in_width;
        constants.in_dims[1] = in_height;
        constants.crop_dims[0] = 512;
        constants.crop_dims[1] = 512;
        constants.out_dims[0] = out_width_;
        constants.out_dims[1] = out_height_;
        vkCmdPushConstants(cmd, compute_pipeline_->GetPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(CropResizePushConstants), &constants);

        // d. Dispatch compute job
        vkCmdDispatch(cmd, (out_width_ + 7) / 8, (out_height_ + 7) / 8, 1);

        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query_idx_base + 1);
        }

        // e. Add barrier (Shader Write -> Transfer Read)
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 
                             0, 1, &barrier, 0, nullptr, 0, nullptr);

        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, query_pool, query_idx_base + 2);
        }

        // f. Copy device-local buffer to host-visible buffer
        VkBufferCopy copy_region = {};
        copy_region.size = out_size_bytes_;
        vkCmdCopyBuffer(cmd, output_buffers_device_[buffer_index],
                        readback_buffers_[buffer_index], 1, &copy_region);

        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, query_pool, query_idx_base + 3);
        }

        // g. Add barrier (Transfer Write -> Host Read)
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT; 
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 
                             0, 1, &barrier, 0, nullptr, 0, nullptr);

        // h. End command buffer
        vkEndCommandBuffer(cmd);

        // i. Submit commands
        VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;
        if (vkQueueSubmit(context_->GetComputeQueue(), 1, &submit_info, fences_[buffer_index]) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to submit command buffer!");
        }
        
        // --- DO NOT WAIT HERE ---

    } catch (const std::exception& e) {
        std::cerr << "[Debug SubmitPreprocessImage-AHB] EXCEPTION CAUGHT: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}
#endif  // __ANDROID__

// --- [ADDED] SyncPreprocess ---
bool VulkanImageProcessor::SyncPreprocess(void* out_data, int buffer_index) {
    if (!context_) return false;
    VkDevice device = context_->GetDevice();

    std::chrono::high_resolution_clock::time_point start_time, end_time;

    try {
        // --- 1. Wait for the work to complete ---
        // This is the main "sync" point.
        vkWaitForFences(device, 1, &fences_[buffer_index], VK_TRUE, UINT64_MAX);
        
        // --- 2. Get timestamp results ---
        VkQueryPool query_pool = context_->GetQueryPool();
        float timestamp_period_ns = context_->GetTimestampPeriod();
        bool can_query_timestamps = (query_pool != VK_NULL_HANDLE && timestamp_period_ns > 0.0f);

        if (can_query_timestamps) {
            uint32_t query_idx_base = buffer_index * 4;
            uint64_t timestamps[4] = {0};
            VkResult result = vkGetQueryPoolResults(
                device, query_pool, query_idx_base, 4,
                sizeof(uint64_t) * 4, timestamps,
                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

            if (result == VK_SUCCESS) { 
                last_timings_[buffer_index].gpu_shader_ms = 
                    (timestamps[1] - timestamps[0]) * timestamp_period_ns / 1000000.0;
                last_timings_[buffer_index].gpu_readback_ms = 
                    (timestamps[3] - timestamps[2]) * timestamp_period_ns / 1000000.0;
            } else {
                 std::cerr << "vkGetQueryPoolResults failed with code: " << result << std::endl;
            }
        }

        // --- 3. Readback Data ---
        start_time = std::chrono::high_resolution_clock::now();
        void* mapped_data = VulkanUtils::MapBufferMemory(
            device, readback_buffers_memory_[buffer_index], out_size_bytes_);
        memcpy(out_data, mapped_data, out_size_bytes_);
        VulkanUtils::UnmapBufferMemory(device, readback_buffers_memory_[buffer_index]);
        end_time = std::chrono::high_resolution_clock::now();
        last_timings_[buffer_index].readback_copy_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();

    } catch (const std::exception& e) {
        std::cerr << "[Debug SyncPreprocess] EXCEPTION CAUGHT: " << e.what() << std::endl;
        return false;
    }

    return true;
}


// --- [OMITTED] PreprocessImage_ZeroCopy (unchanged) ---
bool VulkanImageProcessor::PreprocessImage_ZeroCopy(AHardwareBuffer* in_buffer, int in_width,
                                                    int in_height) {
    std::cerr << "PreprocessImage_ZeroCopy is not implemented." << std::endl;
    return false;
}

// --- [MODIFIED] createPersistentResources ---
bool VulkanImageProcessor::createPersistentResources() {
    VkDevice device = context_->GetDevice();
    VkPhysicalDevice physical_device = context_->GetPhysicalDevice();
    try {
        // --- Create persistent input image (single) ---
        std::cout << "[Debug] Creating persistent input image..." << std::endl;
        VulkanUtils::CreateImage(
            device, physical_device, max_in_width_, max_in_height_, in_image_format_,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, in_image_, in_image_memory_);
        in_image_view_ = VulkanUtils::CreateImageView(device, in_image_, in_image_format_);
        std::cout << "[Debug] Persistent input image/view created." << std::endl;

        // --- Create command buffers ---
        VkCommandBufferAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc_info.commandPool = context_->GetCommandPool();
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = kMaxFramesInFlight;
        if (vkAllocateCommandBuffers(device, &alloc_info, command_buffers_.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffers!");
        }

        // --- Create double-buffered resources ---
        for (int i = 0; i < kMaxFramesInFlight; ++i) {
            std::cout << "[Debug] Creating resources for frame " << i << "..." << std::endl;
            // Staging buffer
            VulkanUtils::CreateBuffer(
                device, physical_device, in_staging_size_bytes_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                staging_buffers_[i], staging_buffers_memory_[i]);
            
            // Device-local output buffer
            VulkanUtils::CreateBuffer(
                device, physical_device, out_size_bytes_,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, output_buffers_device_[i],
                output_buffers_device_memory_[i]);
            
            // Host-visible readback buffer
            VulkanUtils::CreateBuffer(device, physical_device, out_size_bytes_,
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      readback_buffers_[i], readback_buffers_memory_[i]);
            
            // Fence (create in signaled state so first wait passes)
            VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT; 
            if (vkCreateFence(device, &fence_info, nullptr, &fences_[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create fence!");
            }
        }
        std::cout << "[Debug] Double-buffered resources created." << std::endl;

        if (!createDescriptorPool()) {
            throw std::runtime_error("Failed to create descriptor pool.");
        }
        std::cout << "[Debug] Allocating descriptor sets..." << std::endl;
        if (!createDescriptorSets()) { // Plural
            throw std::runtime_error("Failed to create descriptor sets.");
        }
        std::cout << "[Debug] Descriptor sets allocated and updated." << std::endl;
        
        std::cout << "[Debug] Transitioning persistent input image to GENERAL..." << std::endl;
        VkCommandBuffer cmd = context_->BeginOneTimeCommands();
        VulkanUtils::TransitionImageLayout(cmd, in_image_, in_image_format_,
                                           VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        context_->EndAndSubmitCommands(cmd);
        std::cout << "[Debug] Persistent input image transitioned." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to create persistent resources: " << e.what() << std::endl;
        return false;
    }
    return true;
}

// --- [OMITTED] destroyAhbInputResources (unchanged) ---
void VulkanImageProcessor::destroyAhbInputResources() {
    if (!context_ || !context_->GetDevice()) {
        return;
    }
    VkDevice device = context_->GetDevice();
    if (ahb_in_image_view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device, ahb_in_image_view_, nullptr);
        ahb_in_image_view_ = VK_NULL_HANDLE;
    }
    if (ahb_in_image_ != VK_NULL_HANDLE) {
        vkDestroyImage(device, ahb_in_image_, nullptr);
        ahb_in_image_ = VK_NULL_HANDLE;
    }
    if (ahb_in_image_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device, ahb_in_image_memory_, nullptr);
        ahb_in_image_memory_ = VK_NULL_HANDLE;
    }
    last_in_ahb_ = nullptr;
}

// --- [MODIFIED] destroyPersistentResources ---
void VulkanImageProcessor::destroyPersistentResources() {
    if (!context_ || !context_->GetDevice()) {
        return;
    }
    VkDevice device = context_->GetDevice();

    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
        // No need to free sets, pool deletion does that
        descriptor_sets_.clear(); 
    }
    
    // Destroy command buffers
    if (!command_buffers_.empty() && command_buffers_[0] != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device, context_->GetCommandPool(),
                             static_cast<uint32_t>(command_buffers_.size()),
                             command_buffers_.data());
        command_buffers_.clear();
    }

    // Destroy single input image
    if (in_image_view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device, in_image_view_, nullptr);
        in_image_view_ = VK_NULL_HANDLE;
    }
    if (in_image_ != VK_NULL_HANDLE) {
        vkDestroyImage(device, in_image_, nullptr);
        in_image_ = VK_NULL_HANDLE;
    }
    if (in_image_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device, in_image_memory_, nullptr);
        in_image_memory_ = VK_NULL_HANDLE;
    }

    // Destroy double-buffered resources
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        if (i >= fences_.size()) break; // Safeguard if init failed

        if (fences_[i] != VK_NULL_HANDLE) {
            vkDestroyFence(device, fences_[i], nullptr);
        }
        if (staging_buffers_[i] != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, staging_buffers_[i], nullptr);
        }
        if (staging_buffers_memory_[i] != VK_NULL_HANDLE) {
            vkFreeMemory(device, staging_buffers_memory_[i], nullptr);
        }
        if (output_buffers_device_[i] != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, output_buffers_device_[i], nullptr);
        }
        if (output_buffers_device_memory_[i] != VK_NULL_HANDLE) {
            vkFreeMemory(device, output_buffers_device_memory_[i], nullptr);
        }
        if (readback_buffers_[i] != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, readback_buffers_[i], nullptr);
        }
        if (readback_buffers_memory_[i] != VK_NULL_HANDLE) {
            vkFreeMemory(device, readback_buffers_memory_[i], nullptr);
        }
    }
    fences_.clear();
    staging_buffers_.clear();
    staging_buffers_memory_.clear();
    output_buffers_device_.clear();
    output_buffers_device_memory_.clear();
    readback_buffers_.clear();
    readback_buffers_memory_.clear();

    destroyAhbInputResources();
}

// --- [MODIFIED] createDescriptorPool ---
bool VulkanImageProcessor::createDescriptorPool() {
    VkDevice device = context_->GetDevice();
    // Pool needs to be large enough for kMaxFramesInFlight sets
    VkDescriptorPoolSize pool_size_storage_image = {};
    pool_size_storage_image.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_size_storage_image.descriptorCount = kMaxFramesInFlight; // 1 per set
    VkDescriptorPoolSize pool_size_storage_buffer = {};
    pool_size_storage_buffer.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size_storage_buffer.descriptorCount = kMaxFramesInFlight; // 1 per set
    
    std::vector<VkDescriptorPoolSize> pool_sizes = {pool_size_storage_image,
                                                    pool_size_storage_buffer};
    VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = kMaxFramesInFlight; // Max sets we can allocate
    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        return false;
    }
    return true;
}

// --- [MODIFIED] createDescriptorSets (plural) ---
bool VulkanImageProcessor::createDescriptorSets() {
    VkDevice device = context_->GetDevice();
    
    std::vector<VkDescriptorSetLayout> layouts(kMaxFramesInFlight,
                                             compute_pipeline_->GetDescriptorSetLayout());
    
    VkDescriptorSetAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = kMaxFramesInFlight;
    alloc_info.pSetLayouts = layouts.data();
    
    if (vkAllocateDescriptorSets(device, &alloc_info, descriptor_sets_.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor sets!");
    }

    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        VkDescriptorImageInfo input_image_info = {};
        input_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;  
        input_image_info.imageView = in_image_view_; // All sets use the *same* input image
        input_image_info.sampler = VK_NULL_HANDLE;

        VkWriteDescriptorSet write_input = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write_input.dstSet = descriptor_sets_[i];
        write_input.dstBinding = 0;
        write_input.dstArrayElement = 0;
        write_input.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  
        write_input.descriptorCount = 1;
        write_input.pImageInfo = &input_image_info;

        VkDescriptorBufferInfo output_buffer_info = {};
        output_buffer_info.buffer = output_buffers_device_[i]; // This is unique
        output_buffer_info.offset = 0;
        output_buffer_info.range = out_size_bytes_;

        VkWriteDescriptorSet write_output = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write_output.dstSet = descriptor_sets_[i];
        write_output.dstBinding = 1;
        write_output.dstArrayElement = 0;
        write_output.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write_output.descriptorCount = 1;
        write_output.pBufferInfo = &output_buffer_info;

        std::vector<VkWriteDescriptorSet> descriptor_writes = {write_input, write_output};
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptor_writes.size()),
                               descriptor_writes.data(), 0, nullptr);
    }
    
    std::cout << "[Debug] " << kMaxFramesInFlight << " descriptor sets updated." << std::endl;
    return true;
}