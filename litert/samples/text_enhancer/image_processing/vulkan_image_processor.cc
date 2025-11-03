#include "litert/samples/text_enhancer/image_processing/vulkan_image_processor.h"
#include "vulkan/vulkan_utils.h" // Our new utils header

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
// Note: vulkan_android.h is included by vulkan_image_processor.h
#endif

VulkanImageProcessor::VulkanImageProcessor() {
    // All members initialized in header
}

VulkanImageProcessor::~VulkanImageProcessor() {
    Shutdown();
}

bool VulkanImageProcessor::Initialize(const std::string& shader_spirv_path,
                                      int out_width, int out_height) {
    try {
        out_width_ = out_width;
        out_height_ = out_height;
        // 4 channels (RGBA) of float (4 bytes)
        out_size_bytes_ = static_cast<VkDeviceSize>(out_width) * out_height * 4 * sizeof(float);

        // 1. Initialize core context
        context_ = std::make_unique<VulkanContext>();
        if (!context_->Initialize()) {
            throw std::runtime_error("Failed to initialize VulkanContext.");
        }

        #ifdef __ANDROID__
        // Load the AHB extension function pointer
        vkGetAndroidHardwareBufferPropertiesANDROID_ = (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)vkGetDeviceProcAddr(context_->GetDevice(), "vkGetAndroidHardwareBufferPropertiesANDROID");
        if (!vkGetAndroidHardwareBufferPropertiesANDROID_) {
            throw std::runtime_error("Failed to get vkGetAndroidHardwareBufferPropertiesANDROID proc addr.");
        }
        std::cout << "Loaded vkGetAndroidHardwareBufferPropertiesANDROID." << std::endl;
        #endif

        // 2. Initialize compute pipeline
        compute_pipeline_ = std::make_unique<VulkanComputePipeline>();
        if (!compute_pipeline_->Initialize(context_.get(), shader_spirv_path)) {
            throw std::runtime_error("Failed to initialize VulkanComputePipeline.");
        }

        // 3. Create resources that are re-used every frame
        if (!createPersistentResources()) {
            throw std::runtime_error("Failed to create persistent resources.");
        }

    } catch (const std::exception& e) {
        std::cerr << "VulkanImageProcessor Initialization Error: " << e.what() << std::endl;
        Shutdown(); // Clean up partial initialization
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
bool VulkanImageProcessor::PreprocessImage(const unsigned char* in_data,
                                           int in_width, int in_height, int in_channels,
                                           float* out_data) {
    // --- LOG RESTORED ---
    std::cout << "[Debug PreprocessImage] Entering function." << std::endl;
    // --- END LOG ---

    // Check if initialized
    if (!context_ || !compute_pipeline_) {
        std::cerr << "Processor not initialized." << std::endl;
        return false;
    }
    VkDevice device = context_->GetDevice();

    if (in_channels != 4) {
        std::cerr << "Vulkan processor currently only supports 4-channel RGBA images." << std::endl;
        return false;
    }

    const VkDeviceSize in_size_bytes = static_cast<VkDeviceSize>(in_width) * in_height * in_channels;
    const VkFormat in_image_format = VK_FORMAT_R8G8B8A8_UNORM;

    // --- LOG RESTORED ---
    std::cout << "[Debug PreprocessImage] Input size: " << in_size_bytes << " bytes, Output size: " << out_size_bytes_ << " bytes." << std::endl;
    // --- END LOG ---

    // --- 1. Create DYNAMIC (per-frame) Resources ---
    // These resources change size, so they must be created/destroyed each time.
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_buffer_memory = VK_NULL_HANDLE;
    VkImage in_image = VK_NULL_HANDLE;
    VkDeviceMemory in_image_memory = VK_NULL_HANDLE;
    VkImageView in_image_view = VK_NULL_HANDLE;

    // --- LOG RESTORED ---
    std::cout << "[Debug PreprocessImage] Handles initialized to VK_NULL_HANDLE." << std::endl;
    // --- END LOG ---

    try {
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Entering try block." << std::endl;
        // --- END LOG ---

        // a. Create staging buffer (Host -> Device)
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Creating staging buffer..." << std::endl;
        // --- END LOG ---
        VulkanUtils::CreateBuffer(device, context_->GetPhysicalDevice(), in_size_bytes,
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  staging_buffer, staging_buffer_memory);
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Staging buffer created." << std::endl;
        // --- END LOG ---

        // b. Create input image (Device-local, sampled by shader)
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Creating input image (" << in_width << "x" << in_height << ")..." << std::endl;
        // --- END LOG ---
        VulkanUtils::CreateImage(device, context_->GetPhysicalDevice(), in_width, in_height, in_image_format,
                                 VK_IMAGE_TILING_OPTIMAL,
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                 in_image, in_image_memory);
        in_image_view = VulkanUtils::CreateImageView(device, in_image, in_image_format);
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Input image created." << std::endl;
        // --- END LOG ---

        // --- 2. Copy Data to Staging Buffer ---
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Mapping staging buffer..." << std::endl;
        // --- END LOG ---
        void* mapped_data = VulkanUtils::MapBufferMemory(device, staging_buffer_memory, in_size_bytes);
        memcpy(mapped_data, in_data, in_size_bytes);
        VulkanUtils::UnmapBufferMemory(device, staging_buffer_memory);
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Data copied to staging buffer." << std::endl;
        // --- END LOG ---

        // --- 3. Update Persistent Descriptor Set ---
        // We re-use the *persistent* descriptor set, just update it to point
        // to the *new* input image view.
        VkDescriptorImageInfo input_image_info = {};
        input_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        input_image_info.imageView = in_image_view;
        input_image_info.sampler = sampler_; // Use persistent sampler

        VkWriteDescriptorSet write_input = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write_input.dstSet = descriptor_set_;
        write_input.dstBinding = 0; // Binding 0 (sampler2D)
        write_input.dstArrayElement = 0;
        write_input.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_input.descriptorCount = 1;
        write_input.pImageInfo = &input_image_info;
        
        // Note: Binding 1 (output image) is persistent, so it was set
        // at initialization and does not need to be updated.
        vkUpdateDescriptorSets(device, 1, &write_input, 0, nullptr);
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Descriptor set updated." << std::endl;
        // --- END LOG ---

        // --- 4. Record & Submit Commands ---
        vkResetFences(device, 1, &fence_);
        VkCommandBuffer cmd = context_->BeginOneTimeCommands(); // This will log "Beginning one-time commands..."

        // a. Copy staging buffer to input image
        VulkanUtils::TransitionImageLayout(cmd, in_image, in_image_format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VulkanUtils::CopyBufferToImage(cmd, staging_buffer, in_image, in_width, in_height);
        
        // b. Transition layouts for compute shader
        VulkanUtils::TransitionImageLayout(cmd, in_image, in_image_format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        VulkanUtils::TransitionImageLayout(cmd, out_image_, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL); // for imageStore

        // c. Bind pipeline and descriptors
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_->GetPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_->GetPipelineLayout(), 0, 1, &descriptor_set_, 0, nullptr);

        // d. Set push constants
        CropResizePushConstants constants = {};
        constants.in_dims[0]   = in_width;
        constants.in_dims[1]   = in_height;
        constants.crop_dims[0] = 512; // As requested in original code
        constants.crop_dims[1] = 512;
        constants.out_dims[0]  = out_width_;
        constants.out_dims[1]  = out_height_;
        vkCmdPushConstants(cmd, compute_pipeline_->GetPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CropResizePushConstants), &constants);

        // e. Dispatch compute job
        vkCmdDispatch(cmd, (out_width_ + 7) / 8, (out_height_ + 7) / 8, 1);

        // f. Add barrier to make sure shader writes are finished
        VkMemoryBarrier barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);

        // g. Copy output image to readback buffer
        VulkanUtils::TransitionImageLayout(cmd, out_image_, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VulkanUtils::CopyImageToBuffer(cmd, out_image_, readback_buffer_, out_width_, out_height_);

        // h. End command buffer
        vkEndCommandBuffer(cmd);

        // i. Submit commands
        VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Submitting commands..." << std::endl;
        // --- END LOG ---
        if (vkQueueSubmit(context_->GetComputeQueue(), 1, &submit_info, fence_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit command buffer!");
        }

        // Wait for operations to complete
        vkWaitForFences(device, 1, &fence_, VK_TRUE, UINT64_MAX);
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Commands submitted and awaited." << std::endl;
        // --- END LOG ---
        vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);

        // --- 5. Readback Data ---
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Mapping readback buffer..." << std::endl;
        // --- END LOG ---
        mapped_data = VulkanUtils::MapBufferMemory(device, readback_buffer_memory_, out_size_bytes_);
        
        // --- LOGS RESTORED ---
        std::cout << "[Debug PreprocessImage] Readback buffer mapped. mapped_data=" 
                  << mapped_data << ", out_data=" << (void*)out_data << std::endl;
        
        if (out_data == nullptr) {
            std::cerr << "[Debug PreprocessImage] FATAL: out_data pointer is NULL." << std::endl;
            throw std::runtime_error("out_data pointer was null before memcpy!");
        }
        if (mapped_data == nullptr) {
            std::cerr << "[Debug PreprocessImage] FATAL: mapped_data pointer is NULL." << std::endl;
            throw std::runtime_error("mapped_data pointer was null before memcpy!");
        }
        
        std::cout << "[Debug PreprocessImage] Calling memcpy..." << std::endl;
        memcpy(out_data, mapped_data, out_size_bytes_);
        std::cout << "[Debug PreprocessImage] memcpy complete." << std::endl;
        // --- END LOGS ---

        VulkanUtils::UnmapBufferMemory(device, readback_buffer_memory_);
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Data read back from GPU." << std::endl;
        // --- END LOG ---

    } catch (const std::exception& e) {
        // --- LOG RESTORED ---
        std::cerr << "[Debug PreprocessImage] EXCEPTION CAUGHT: " << e.what() << std::endl;
        
        // --- 6. Cleanup (on failure) ---
        std::cout << "[Debug PreprocessImage] Cleaning up resources after exception..." << std::endl;
        
        std::cout << "[Debug PreprocessImage] Destroying in_image_view: " << (void*)in_image_view << std::endl;
        if (in_image_view != VK_NULL_HANDLE) vkDestroyImageView(device, in_image_view, nullptr);
        std::cout << "[Debug PreprocessImage] Destroying in_image: " << (void*)in_image << std::endl;
        if (in_image != VK_NULL_HANDLE) vkDestroyImage(device, in_image, nullptr);
        std::cout << "[Debug PreprocessImage] Freeing in_image_memory: " << (void*)in_image_memory << std::endl;
        if (in_image_memory != VK_NULL_HANDLE) vkFreeMemory(device, in_image_memory, nullptr);
        std::cout << "[Debug PreprocessImage] Destroying staging_buffer: " << (void*)staging_buffer << std::endl;
        if (staging_buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, staging_buffer, nullptr);
        std::cout << "[Debug PreprocessImage] Freeing staging_buffer_memory: " << (void*)staging_buffer_memory << std::endl;
        if (staging_buffer_memory != VK_NULL_HANDLE) vkFreeMemory(device, staging_buffer_memory, nullptr);

        std::cout << "[Debug PreprocessImage] Returning false from catch block." << std::endl;
        // --- END LOGS ---
        return false;
    }

    // --- 6. Cleanup (on success) ---
    // --- LOG RESTORED ---
    std::cout << "[Debug PreprocessImage] Cleaning up resources after success..." << std::endl;

    std::cout << "[Debug PreprocessImage] Destroying in_image_view: " << (void*)in_image_view << std::endl;
    vkDestroyImageView(device, in_image_view, nullptr);
    std::cout << "[Debug PreprocessImage] Destroying in_image: " << (void*)in_image << std::endl;
    vkDestroyImage(device, in_image, nullptr);
    std::cout << "[Debug PreprocessImage] Freeing in_image_memory: " << (void*)in_image_memory << std::endl;
    vkFreeMemory(device, in_image_memory, nullptr);
    std::cout << "[Debug PreprocessImage] Destroying staging_buffer: " << (void*)staging_buffer << std::endl;
    vkDestroyBuffer(device, staging_buffer, nullptr);
    std::cout << "[Debug PreprocessImage] Freeing staging_buffer_memory: " << (void*)staging_buffer_memory << std::endl;
    vkFreeMemory(device, staging_buffer_memory, nullptr);

    std::cout << "[Debug PreprocessImage] Returning true from success path." << std::endl;
    // --- END LOGS ---
    return true;
}

#ifdef __ANDROID__
// --- Main Processing Function (AHardwareBuffer Zero-Copy Path) ---
bool VulkanImageProcessor::PreprocessImage(AHardwareBuffer* in_buffer,
                                           int in_width, int in_height,
                                           float* out_data) {
    std::cout << "[Debug PreprocessImage-AHB] Entering function." << std::endl;
    if (!context_ || !compute_pipeline_ || !vkGetAndroidHardwareBufferPropertiesANDROID_) {
        std::cerr << "Processor not initialized or AHB extensions not loaded." << std::endl;
        return false;
    }
    VkDevice device = context_->GetDevice();

    // 1. Import AHB to Vulkan Image
    VkImage in_image = VK_NULL_HANDLE;
    VkDeviceMemory in_image_memory = VK_NULL_HANDLE;
    VkImageView in_image_view = VK_NULL_HANDLE;
    VkFormat in_image_format = VK_FORMAT_UNDEFINED;

    try {
        std::cout << "[Debug PreprocessImage-AHB] Importing AHB..." << std::endl;
        if (!VulkanUtils::ImportAhbToImage(device, context_->GetPhysicalDevice(), in_buffer,
                                            vkGetAndroidHardwareBufferPropertiesANDROID_,
                                            in_image, in_image_memory, in_image_view,
                                            in_image_format)) {
            throw std::runtime_error("Failed to import AHardwareBuffer to VkImage.");
        }
        std::cout << "[Debug PreprocessImage-AHB] AHB imported successfully. Format: " << in_image_format << std::endl;

        // --- 2. Update Persistent Descriptor Set ---
        // We re-use the *persistent* descriptor set, just update it to point
        // to the *new* input image view.
        VkDescriptorImageInfo input_image_info = {};
        input_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        input_image_info.imageView = in_image_view;
        input_image_info.sampler = sampler_; // Use persistent sampler

        VkWriteDescriptorSet write_input = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write_input.dstSet = descriptor_set_;
        write_input.dstBinding = 0; // Binding 0 (sampler2D)
        write_input.dstArrayElement = 0;
        write_input.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_input.descriptorCount = 1;
        write_input.pImageInfo = &input_image_info;
        
        vkUpdateDescriptorSets(device, 1, &write_input, 0, nullptr);
        std::cout << "[Debug PreprocessImage-AHB] Descriptor set updated." << std::endl;

        // --- 3. Record & Submit Commands ---
        vkResetFences(device, 1, &fence_);
        VkCommandBuffer cmd = context_->BeginOneTimeCommands();

        // a. Transition input image layout
        // The image is created as UNDEFINED, transition it for shader read.
        VulkanUtils::TransitionImageLayout(cmd, in_image, in_image_format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        
        // b. Transition output image layout
        VulkanUtils::TransitionImageLayout(cmd, out_image_, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL); // for imageStore

        // c. Bind pipeline and descriptors
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_->GetPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_->GetPipelineLayout(), 0, 1, &descriptor_set_, 0, nullptr);

        // d. Set push constants
        CropResizePushConstants constants = {};
        constants.in_dims[0]   = in_width;
        constants.in_dims[1]   = in_height;
        constants.crop_dims[0] = 512; // As requested in original code
        constants.crop_dims[1] = 512;
        constants.out_dims[0]  = out_width_;
        constants.out_dims[1]  = out_height_;
        vkCmdPushConstants(cmd, compute_pipeline_->GetPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CropResizePushConstants), &constants);

        // e. Dispatch compute job
        vkCmdDispatch(cmd, (out_width_ + 7) / 8, (out_height_ + 7) / 8, 1);

        // f. Add barrier to make sure shader writes are finished
        VkMemoryBarrier barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);

        // g. Copy output image to readback buffer
        VulkanUtils::TransitionImageLayout(cmd, out_image_, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VulkanUtils::CopyImageToBuffer(cmd, out_image_, readback_buffer_, out_width_, out_height_);

        // h. End command buffer
        vkEndCommandBuffer(cmd);

        // i. Submit commands
        VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;
        std::cout << "[Debug PreprocessImage-AHB] Submitting commands..." << std::endl;
        if (vkQueueSubmit(context_->GetComputeQueue(), 1, &submit_info, fence_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit command buffer!");
        }

        // Wait for operations to complete
        vkWaitForFences(device, 1, &fence_, VK_TRUE, UINT64_MAX);
        std::cout << "[Debug PreprocessImage-AHB] Commands submitted and awaited." << std::endl;
        vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);

        // --- 4. Readback Data ---
        std::cout << "[Debug PreprocessImage-AHB] Mapping readback buffer..." << std::endl;
        void* mapped_data = VulkanUtils::MapBufferMemory(device, readback_buffer_memory_, out_size_bytes_);
        
        std::cout << "[Debug PreprocessImage-AHB] Calling memcpy..." << std::endl;
        memcpy(out_data, mapped_data, out_size_bytes_);
        std::cout << "[Debug PreprocessImage-AHB] memcpy complete." << std::endl;

        VulkanUtils::UnmapBufferMemory(device, readback_buffer_memory_);
        std::cout << "[Debug PreprocessImage-AHB] Data read back from GPU." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Debug PreprocessImage-AHB] EXCEPTION CAUGHT: " << e.what() << std::endl;
        // --- 5. Cleanup (on failure) ---
        std::cout << "[Debug PreprocessImage-AHB] Cleaning up resources after exception..." << std::endl;
        if (in_image_view != VK_NULL_HANDLE) vkDestroyImageView(device, in_image_view, nullptr);
        if (in_image != VK_NULL_HANDLE) vkDestroyImage(device, in_image, nullptr);
        if (in_image_memory != VK_NULL_HANDLE) vkFreeMemory(device, in_image_memory, nullptr);
        std::cout << "[Debug PreprocessImage-AHB] Returning false from catch block." << std::endl;
        return false;
    }

    // --- 5. Cleanup (on success) ---
    std::cout << "[Debug PreprocessImage-AHB] Cleaning up resources after success..." << std::endl;
    vkDestroyImageView(device, in_image_view, nullptr);
    vkDestroyImage(device, in_image, nullptr);
    vkFreeMemory(device, in_image_memory, nullptr);
    std::cout << "[Debug PreprocessImage-AHB] Returning true from success path." << std::endl;
    return true;
}
#endif // __ANDROID__


// --- Private Helper Functions Implementation ---

bool VulkanImageProcessor::createPersistentResources() {
    VkDevice device = context_->GetDevice();
    VkPhysicalDevice physical_device = context_->GetPhysicalDevice();
    const VkFormat out_image_format = VK_FORMAT_R32G32B32A32_SFLOAT;

    try {
        // 1. Create readback buffer (persistent)
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Creating readback buffer..." << std::endl;
        VulkanUtils::CreateBuffer(device, physical_device, out_size_bytes_,
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  readback_buffer_, readback_buffer_memory_);
        std::cout << "[Debug PreprocessImage] Readback buffer created." << std::endl;
        // --- END LOGS ---

        // 2. Create output image (persistent)
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Creating output image (" << out_width_ << "x" << out_height_ << ")..." << std::endl;
        VulkanUtils::CreateImage(device, physical_device, out_width_, out_height_, out_image_format,
                                 VK_IMAGE_TILING_OPTIMAL,
                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                 out_image_, out_image_memory_);
        out_image_view_ = VulkanUtils::CreateImageView(device, out_image_, out_image_format);
        std::cout << "[Debug PreprocessImage] Output image created." << std::endl;
        // --- END LOGS ---

        // 3. Create sampler (persistent)
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Creating sampler..." << std::endl;
        sampler_ = VulkanUtils::CreateSampler(device);
        std::cout << "[Debug PreprocessImage] Sampler created." << std::endl;
        // --- END LOGS ---

        // 4. Create descriptor pool & set (persistent)
        if (!createDescriptorPool()) {
            throw std::runtime_error("Failed to create descriptor pool.");
        }
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Allocating descriptor set..." << std::endl;
        // --- END LOG ---
        if (!createDescriptorSet()) { // This function now logs "Descriptor set updated."
            throw std::runtime_error("Failed to create descriptor set.");
        }
        // --- LOG RESTORED ---
        std::cout << "[Debug PreprocessImage] Descriptor set allocated." << std::endl;
        // --- END LOG ---

        // 5. Create fence (persistent)
        VkFenceCreateInfo fence_info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Create signaled for first use
        if (vkCreateFence(device, &fence_info, nullptr, &fence_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fence!");
        }
        // Immediately reset it
        vkResetFences(device, 1, &fence_);

    } catch (const std::exception& e) {
        std::cerr << "Failed to create persistent resources: " << e.what() << std::endl;
        return false;
    }
    return true;
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
    // Descriptor set is freed with pool
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
        descriptor_set_ = VK_NULL_HANDLE; // Was owned by pool
    }
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device, sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }
    if (out_image_view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device, out_image_view_, nullptr);
        out_image_view_ = VK_NULL_HANDLE;
    }
    if (out_image_ != VK_NULL_HANDLE) {
        vkDestroyImage(device, out_image_, nullptr);
        out_image_ = VK_NULL_HANDLE;
    }
    if (out_image_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device, out_image_memory_, nullptr);
        out_image_memory_ = VK_NULL_HANDLE;
    }
    if (readback_buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, readback_buffer_, nullptr);
        readback_buffer_ = VK_NULL_HANDLE;
    }
    if (readback_buffer_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device, readback_buffer_memory_, nullptr);
        readback_buffer_memory_ = VK_NULL_HANDLE;
    }
}

bool VulkanImageProcessor::createDescriptorPool() {
    VkDevice device = context_->GetDevice();
    
    // We need 1 set, with 1 combined sampler and 1 storage image
    VkDescriptorPoolSize pool_size_sampler = {};
    pool_size_sampler.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size_sampler.descriptorCount = 1; // Max sets we can allocate

    VkDescriptorPoolSize pool_size_storage = {};
    pool_size_storage.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_size_storage.descriptorCount = 1;

    std::vector<VkDescriptorPoolSize> pool_sizes = {pool_size_sampler, pool_size_storage};

    VkDescriptorPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = 1; // Max number of descriptor sets

    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        return false;
    }
    return true;
}

bool VulkanImageProcessor::createDescriptorSet() {
    VkDevice device = context_->GetDevice();
    
    VkDescriptorSetAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    
    // --- FIX from previous turn ---
    VkDescriptorSetLayout layout = compute_pipeline_->GetDescriptorSetLayout();
    alloc_info.pSetLayouts = &layout;
    // --- END FIX ---

    if (vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set!");
    }

    // --- Update Binding 1 (Output Image) ---
    // Binding 0 (Input) will be updated in PreprocessImage
    VkDescriptorImageInfo output_image_info = {};
    output_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    output_image_info.imageView = out_image_view_;
    
    VkWriteDescriptorSet write_output = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write_output.dstSet = descriptor_set_;
    write_output.dstBinding = 1; // Binding 1 (image2D)
    write_output.dstArrayElement = 0;
    write_output.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write_output.descriptorCount = 1;
    write_output.pImageInfo = &output_image_info;

    vkUpdateDescriptorSets(device, 1, &write_output, 0, nullptr);
    
    // --- LOG RESTORED ---
    std::cout << "[Debug PreprocessImage] Descriptor set updated." << std::endl;
    // --- END LOG ---
    return true;
}