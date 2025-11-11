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
// [OMITTED FOR BREVITY - NO CHANGES]
VulkanImageProcessor::VulkanImageProcessor() {
    // All members initialized in header
}
VulkanImageProcessor::~VulkanImageProcessor() { Shutdown(); }
bool VulkanImageProcessor::Initialize(const std::string& shader_spirv_path, int max_in_width,
                                      int max_in_height, int max_in_channels, int out_width,
                                      int out_height, bool is_output_int8) {
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


// --- Main Processing Function (Staging Buffer Path) ---
bool VulkanImageProcessor::PreprocessImage(const unsigned char* in_data, int in_width,
                                           int in_height, int in_channels, void* out_data) {
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

    // --- Reset timings for this run ---
    last_timings_ = {};
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    
    // --- NEW: Get context extras ---
    VkQueryPool query_pool = context_->GetQueryPool();
    float timestamp_period_ns = context_->GetTimestampPeriod();
    bool can_query_timestamps = (query_pool != VK_NULL_HANDLE && timestamp_period_ns > 0.0f);
    // --- END NEW ---

    try {
        // --- 1. Copy Data to *Persistent* Staging Buffer ---
        start_time = std::chrono::high_resolution_clock::now();
        void* mapped_data =
            VulkanUtils::MapBufferMemory(device, staging_buffer_memory_, in_size_bytes);
        memcpy(mapped_data, in_data, in_size_bytes);
        VulkanUtils::UnmapBufferMemory(device, staging_buffer_memory_);
        end_time = std::chrono::high_resolution_clock::now();
        last_timings_.staging_copy_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();

        // --- 2. NO Descriptor Set Update (Done at init) ---

        // --- 3. Record & Submit Commands ---
        start_time = std::chrono::high_resolution_clock::now(); // Start CPU timer
        vkResetFences(device, 1, &fence_);
        VkCommandBuffer cmd = context_->BeginOneTimeCommands();

        // --- NEW: Reset queries ---
        if (can_query_timestamps) {
            vkCmdResetQueryPool(cmd, query_pool, 0, 4);
        }

        // a. Copy *persistent* staging buffer to *persistent* input image
        VulkanUtils::TransitionImageLayout(cmd, in_image_, in_image_format_,
                                           VK_IMAGE_LAYOUT_GENERAL,
                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VulkanUtils::CopyBufferToImage(cmd, staging_buffer_, in_image_, in_width, in_height);

        // b. Transition *persistent* input image for shader read
        VulkanUtils::TransitionImageLayout(cmd, in_image_, in_image_format_,
                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                           VK_IMAGE_LAYOUT_GENERAL);

        // --- NEW: Timestamp before shader ---
        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, 0);
        }

        // c. Bind pipeline and descriptors
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_->GetPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compute_pipeline_->GetPipelineLayout(), 0, 1, &descriptor_set_, 0,
                                nullptr);

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

        // --- NEW: Timestamp after shader ---
        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, 1);
        }

        // f. Barrier: Shader writes -> Transfer read
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                             nullptr);

        // --- NEW: Timestamp before copy ---
        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, query_pool, 2);
        }

        // g. Copy device-local buffer to host-visible buffer
        VkBufferCopy copy_region = {};
        copy_region.size = out_size_bytes_;
        vkCmdCopyBuffer(cmd, output_buffer_device_, readback_buffer_, 1, &copy_region);

        // --- NEW: Timestamp after copy ---
        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, query_pool, 3);
        }
        
        // h. Barrier: Transfer write -> Host read
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

        if (vkQueueSubmit(context_->GetComputeQueue(), 1, &submit_info, fence_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit command buffer!");
        }

        // Wait for operations to complete
        vkWaitForFences(device, 1, &fence_, VK_TRUE, UINT64_MAX);
        end_time = std::chrono::high_resolution_clock::now(); // End CPU timer
        last_timings_.gpu_submit_wait_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);

        // --- NEW: Get timestamp results ---
        if (can_query_timestamps) {
            uint64_t timestamps[4] = {0};
            VkResult result = vkGetQueryPoolResults(
                device, query_pool, 0, 4,
                sizeof(uint64_t) * 4, timestamps,
                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

            if (result == VK_SUCCESS) { 
                last_timings_.gpu_shader_ms = 
                    (timestamps[1] - timestamps[0]) * timestamp_period_ns / 1000000.0;
                last_timings_.gpu_readback_ms = 
                    (timestamps[3] - timestamps[2]) * timestamp_period_ns / 1000000.0;
            } else {
                 std::cerr << "vkGetQueryPoolResults failed with code: " << result << std::endl;
            }
        }
        // --- END NEW ---

        // --- 4. Readback Data ---
        start_time = std::chrono::high_resolution_clock::now();
        mapped_data =
            VulkanUtils::MapBufferMemory(device, readback_buffer_memory_, out_size_bytes_);

        if (out_data == nullptr || mapped_data == nullptr) {
            throw std::runtime_error("out_data or mapped_data pointer was null before memcpy!");
        }
        
        memcpy(out_data, mapped_data, out_size_bytes_);

        VulkanUtils::UnmapBufferMemory(device, readback_buffer_memory_);
        end_time = std::chrono::high_resolution_clock::now();
        last_timings_.readback_copy_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();

    } catch (const std::exception& e) {
        std::cerr << "[Debug PreprocessImage] EXCEPTION CAUGHT: " << e.what() << std::endl;
        return false;
    }

    return true;
}

#ifdef __ANDROID__
// --- Main Processing Function (AHardwareBuffer Path) ---
bool VulkanImageProcessor::PreprocessImage(AHardwareBuffer* in_buffer, int in_width, int in_height,
                                           void* out_data) {
    if (!context_ || !compute_pipeline_ || !vkGetAndroidHardwareBufferPropertiesANDROID_ ||
        !vkGetMemoryAndroidHardwareBufferANDROID_) {
        std::cerr << "Processor not initialized or AHB extensions not loaded." << std::endl;
        return false;
    }
    VkDevice device = context_->GetDevice();

    // --- Reset timings for this run ---
    last_timings_ = {};
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::chrono::high_resolution_clock::time_point gpu_start_time, gpu_end_time;
    
    // --- NEW: Get context extras ---
    VkQueryPool query_pool = context_->GetQueryPool();
    float timestamp_period_ns = context_->GetTimestampPeriod();
    bool can_query_timestamps = (query_pool != VK_NULL_HANDLE && timestamp_period_ns > 0.0f);
    // --- END NEW ---
    
    VkFormat in_image_format = VK_FORMAT_UNDEFINED;

    try {
        start_time = std::chrono::high_resolution_clock::now();

        // --- Caching Logic ---
        if (in_buffer != last_in_ahb_ || ahb_in_image_ == VK_NULL_HANDLE) {
            std::cout << "[Debug PreprocessImage-AHB] New AHB handle detected. Caching..." << std::endl;
            vkDeviceWaitIdle(device);
            destroyAhbInputResources();

            // 1. Import AHB
            if (!VulkanUtils::ImportAhbToImage(
                    device, context_->GetPhysicalDevice(), in_buffer,
                    vkGetAndroidHardwareBufferPropertiesANDROID_,
                    vkGetMemoryAndroidHardwareBufferANDROID_,
                    VK_IMAGE_USAGE_STORAGE_BIT,
                    ahb_in_image_, ahb_in_image_memory_, ahb_in_image_view_, in_image_format)) {
                throw std::runtime_error("Failed to import AHardwareBuffer to VkImage.");
            }

            // 2. Update Descriptor Set
            VkDescriptorImageInfo input_image_info = {};
            input_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            input_image_info.imageView = ahb_in_image_view_;
            input_image_info.sampler = VK_NULL_HANDLE;

            VkWriteDescriptorSet write_input = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write_input.dstSet = descriptor_set_;
            write_input.dstBinding = 0; 
            write_input.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_input.descriptorCount = 1;
            write_input.pImageInfo = &input_image_info;

            vkUpdateDescriptorSets(device, 1, &write_input, 0, nullptr);
            
            // 3. Transition layout
            VkCommandBuffer cmd = context_->BeginOneTimeCommands();
            VulkanUtils::TransitionImageLayout(cmd, ahb_in_image_, in_image_format,
                                            VK_IMAGE_LAYOUT_UNDEFINED,
                                            VK_IMAGE_LAYOUT_GENERAL);
            context_->EndAndSubmitCommands(cmd); 

            // 4. Save handle
            last_in_ahb_ = in_buffer;
            std::cout << "[Debug PreprocessImage-AHB] Caching complete." << std::endl;
        }
        // --- END Caching Logic ---

        end_time = std::chrono::high_resolution_clock::now();
        last_timings_.staging_copy_ms = 
            std::chrono::duration<double, std::milli>(end_time - start_time).count();


        // --- 3. Record & Submit Commands ---
        gpu_start_time = std::chrono::high_resolution_clock::now(); // Start CPU timer
        vkResetFences(device, 1, &fence_);
        VkCommandBuffer cmd = context_->BeginOneTimeCommands();

        // --- NEW: Reset queries ---
        if (can_query_timestamps) {
            vkCmdResetQueryPool(cmd, query_pool, 0, 4);
        }

        // a. Transition (already in GENERAL)

        // --- NEW: Timestamp before shader ---
        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, 0);
        }

        // b. Bind pipeline and descriptors
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_->GetPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compute_pipeline_->GetPipelineLayout(), 0, 1, &descriptor_set_, 0,
                                nullptr);

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

        // --- NEW: Timestamp after shader ---
        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, 1);
        }

        // e. Add barrier (Shader Write -> Transfer Read)
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 
                             0, 1, &barrier, 0, nullptr, 0, nullptr);

        // --- NEW: Timestamp before copy ---
        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, query_pool, 2);
        }

        // f. Copy device-local buffer to host-visible buffer
        VkBufferCopy copy_region = {};
        copy_region.size = out_size_bytes_;
        vkCmdCopyBuffer(cmd, output_buffer_device_, readback_buffer_, 1, &copy_region);

        // --- NEW: Timestamp after copy ---
        if (can_query_timestamps) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, query_pool, 3);
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
        if (vkQueueSubmit(context_->GetComputeQueue(), 1, &submit_info, fence_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit command buffer!");
        }

        vkWaitForFences(device, 1, &fence_, VK_TRUE, UINT64_MAX);
        // --- END GPU WAIT ---
        
        gpu_end_time = std::chrono::high_resolution_clock::now(); // End GPU timer
        last_timings_.gpu_submit_wait_ms =
            std::chrono::duration<double, std::milli>(gpu_end_time - gpu_start_time).count();
        
        vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);


        // --- NEW: Get timestamp results ---
        if (can_query_timestamps) {
            uint64_t timestamps[4] = {0};
            VkResult result = vkGetQueryPoolResults(
                device, query_pool, 0, 4,
                sizeof(uint64_t) * 4, timestamps,
                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

            if (result == VK_SUCCESS) {
                last_timings_.gpu_shader_ms = 
                    (timestamps[1] - timestamps[0]) * timestamp_period_ns / 1000000.0;
                last_timings_.gpu_readback_ms = 
                    (timestamps[3] - timestamps[2]) * timestamp_period_ns / 1000000.0;
            } else {
                 std::cerr << "vkGetQueryPoolResults failed with code: " << result << std::endl;
            }
        }
        // --- END NEW ---


        // --- 4. Readback Data ---
        start_time = std::chrono::high_resolution_clock::now();
        void* mapped_data =
            VulkanUtils::MapBufferMemory(device, readback_buffer_memory_, out_size_bytes_);
        memcpy(out_data, mapped_data, out_size_bytes_);
        VulkanUtils::UnmapBufferMemory(device, readback_buffer_memory_);
        end_time = std::chrono::high_resolution_clock::now();
        last_timings_.readback_copy_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();

    } catch (const std::exception& e) {
        std::cerr << "[Debug PreprocessImage-AHB] EXCEPTION CAUGHT: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}
#endif  // __ANDROID__

// --- PreprocessImage_ZeroCopy, createPersistentResources, destroyAhbInputResources, destroyPersistentResources, createDescriptorPool, createDescriptorSet ---
// [OMITTED FOR BREVITY - NO CHANGES IN THESE FUNCTIONS]
bool VulkanImageProcessor::PreprocessImage_ZeroCopy(AHardwareBuffer* in_buffer, int in_width,
                                                    int in_height) {
    std::cerr << "PreprocessImage_ZeroCopy is not implemented." << std::endl;
    return false;
}
bool VulkanImageProcessor::createPersistentResources() {
    VkDevice device = context_->GetDevice();
    VkPhysicalDevice physical_device = context_->GetPhysicalDevice();
    try {
        std::cout << "[Debug] Creating persistent staging buffer..." << std::endl;
        VulkanUtils::CreateBuffer(
            device, physical_device, in_staging_size_bytes_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging_buffer_, staging_buffer_memory_);
        std::cout << "[Debug] Creating persistent input image..." << std::endl;
        VulkanUtils::CreateImage(
            device, physical_device, max_in_width_, max_in_height_, in_image_format_,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, in_image_, in_image_memory_);
        in_image_view_ = VulkanUtils::CreateImageView(device, in_image_, in_image_format_);
        std::cout << "[Debug] Persistent input image/view created." << std::endl;
        std::cout << "[Debug] Creating device-local output buffer..." << std::endl;
        VulkanUtils::CreateBuffer(
            device, physical_device, out_size_bytes_,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, output_buffer_device_,
            output_buffer_device_memory_);
        std::cout << "[Debug] Creating host-visible readback buffer..." << std::endl;
        VulkanUtils::CreateBuffer(device, physical_device, out_size_bytes_,
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  readback_buffer_, readback_buffer_memory_);
        std::cout << "[Debug] Output/readback buffers created." << std::endl;
        if (!createDescriptorPool()) {
            throw std::runtime_error("Failed to create descriptor pool.");
        }
        std::cout << "[Debug] Allocating descriptor set..." << std::endl;
        if (!createDescriptorSet()) {
            throw std::runtime_error("Failed to create descriptor set.");
        }
        std::cout << "[Debug] Descriptor set allocated and updated." << std::endl;
        VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(device, &fence_info, nullptr, &fence_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fence!");
        }
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
void VulkanImageProcessor::destroyPersistentResources() {
    if (!context_ || !context_->GetDevice()) {
        return;
    }
    VkDevice device = context_->GetDevice();
    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device, fence_, nullptr);
        fence_ = VK_NULL_HANDLE;
    }
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
        descriptor_set_ = VK_NULL_HANDLE;
    }
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
    if (staging_buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, staging_buffer_, nullptr);
        staging_buffer_ = VK_NULL_HANDLE;
    }
    if (staging_buffer_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device, staging_buffer_memory_, nullptr);
        staging_buffer_memory_ = VK_NULL_HANDLE;
    }
    if (output_buffer_device_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, output_buffer_device_, nullptr);
        output_buffer_device_ = VK_NULL_HANDLE;
    }
    if (output_buffer_device_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device, output_buffer_device_memory_, nullptr);
        output_buffer_device_memory_ = VK_NULL_HANDLE;
    }
    if (readback_buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, readback_buffer_, nullptr);
        readback_buffer_ = VK_NULL_HANDLE;
    }
    if (readback_buffer_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device, readback_buffer_memory_, nullptr);
        readback_buffer_memory_ = VK_NULL_HANDLE;
    }
    destroyAhbInputResources();
}
bool VulkanImageProcessor::createDescriptorPool() {
    VkDevice device = context_->GetDevice();
    VkDescriptorPoolSize pool_size_storage_image = {};
    pool_size_storage_image.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_size_storage_image.descriptorCount = 1;
    VkDescriptorPoolSize pool_size_storage_buffer = {};
    pool_size_storage_buffer.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size_storage_buffer.descriptorCount = 1;
    std::vector<VkDescriptorPoolSize> pool_sizes = {pool_size_storage_image,
                                                    pool_size_storage_buffer};
    VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = 1;
    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        return false;
    }
    return true;
}
bool VulkanImageProcessor::createDescriptorSet() {
    VkDevice device = context_->GetDevice();
    VkDescriptorSetAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    VkDescriptorSetLayout layout = compute_pipeline_->GetDescriptorSetLayout();
    alloc_info.pSetLayouts = &layout;
    if (vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set!");
    }
    VkDescriptorImageInfo input_image_info = {};
    input_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;  
    input_image_info.imageView = in_image_view_; 
    input_image_info.sampler = VK_NULL_HANDLE;
    VkWriteDescriptorSet write_input = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write_input.dstSet = descriptor_set_;
    write_input.dstBinding = 0;
    write_input.dstArrayElement = 0;
    write_input.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  
    write_input.descriptorCount = 1;
    write_input.pImageInfo = &input_image_info;
    VkDescriptorBufferInfo output_buffer_info = {};
    output_buffer_info.buffer = output_buffer_device_;
    output_buffer_info.offset = 0;
    output_buffer_info.range = out_size_bytes_;
    VkWriteDescriptorSet write_output = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write_output.dstSet = descriptor_set_;
    write_output.dstBinding = 1;
    write_output.dstArrayElement = 0;
    write_output.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_output.descriptorCount = 1;
    write_output.pBufferInfo = &output_buffer_info;
    std::vector<VkWriteDescriptorSet> descriptor_writes = {write_input, write_output};
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptor_writes.size()),
                           descriptor_writes.data(), 0, nullptr);
    std::cout << "[Debug] Descriptor set (Bindings 0 & 1) updated permanently." << std::endl;
    return true;
}