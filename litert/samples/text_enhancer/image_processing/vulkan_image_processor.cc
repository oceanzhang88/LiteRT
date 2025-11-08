#include "litert/samples/text_enhancer/image_processing/vulkan_image_processor.h"

#include <chrono>  // <-- ADDED for benchmarking
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vulkan/vulkan_utils.h"  // Our new utils header

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
// Note: vulkan_android.h is included by vulkan_image_processor.h
#endif

VulkanImageProcessor::VulkanImageProcessor() {
    // All members initialized in header
}

VulkanImageProcessor::~VulkanImageProcessor() { Shutdown(); }

bool VulkanImageProcessor::Initialize(const std::string& shader_spirv_path, int out_width,
                                      int out_height, bool is_output_int8) {
    try {
        out_width_ = out_width;
        out_height_ = out_height;
        is_output_int8_ = is_output_int8;

        if (is_output_int8_) {
            // 3 channels (RGB) of int8_t (1 byte)
            out_size_bytes_ =
                static_cast<VkDeviceSize>(out_width) * out_height * 3 * sizeof(int8_t);
            std::cout << "Vulkan processor configured for INT8 output." << std::endl;
        } else {
            // 3 channels (RGB) of float (4 bytes)
            out_size_bytes_ = static_cast<VkDeviceSize>(out_width) * out_height * 3 * sizeof(float);
            std::cout << "Vulkan processor configured for FLOAT32 output." << std::endl;
        }
        
        std::cout << "Output buffer size: " << out_size_bytes_ << " bytes." << std::endl;

        // 1. Initialize core context
        context_ = std::make_unique<VulkanContext>();
        if (!context_->Initialize()) {
            throw std::runtime_error("Failed to initialize VulkanContext.");
        }

#ifdef __ANDROID__
        // Load the AHB extension function pointer
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

        // 2. Initialize compute pipeline
        //    (This loads the shader_spirv_path you provided)
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
        Shutdown();  // Clean up partial initialization
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
    std::cout << "[Debug PreprocessImage] Entering function." << std::endl;

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
    const VkFormat in_image_format = VK_FORMAT_R8G8B8A8_UNORM;

    std::cout << "[Debug PreprocessImage] Input size: " << in_size_bytes
              << " bytes, Output size: " << out_size_bytes_ << " bytes." << std::endl;

    // --- 1. Create DYNAMIC (per-frame) Resources ---
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_buffer_memory = VK_NULL_HANDLE;
    VkImage in_image = VK_NULL_HANDLE;
    VkDeviceMemory in_image_memory = VK_NULL_HANDLE;
    VkImageView in_image_view = VK_NULL_HANDLE;

    std::cout << "[Debug PreprocessImage] Handles initialized to VK_NULL_HANDLE." << std::endl;

    try {
        std::cout << "[Debug PreprocessImage] Entering try block." << std::endl;

        // a. Create staging buffer (Host -> Device)
        std::cout << "[Debug PreprocessImage] Creating staging buffer..." << std::endl;
        VulkanUtils::CreateBuffer(
            device, context_->GetPhysicalDevice(), in_size_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging_buffer, staging_buffer_memory);

        // b. Create input image (Device-local, read by shader)
        std::cout << "[Debug PreprocessImage] Creating input image (" << in_width << "x"
                  << in_height << ")..." << std::endl;

        VulkanUtils::CreateImage(device, context_->GetPhysicalDevice(), in_width, in_height,
                                 in_image_format, VK_IMAGE_TILING_OPTIMAL,
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, in_image, in_image_memory);

        in_image_view = VulkanUtils::CreateImageView(device, in_image, in_image_format);
        std::cout << "[Debug PreprocessImage] Input image created." << std::endl;

        // --- 2. Copy Data to Staging Buffer ---
        std::cout << "[Debug PreprocessImage] Mapping staging buffer..." << std::endl;
        void* mapped_data =
            VulkanUtils::MapBufferMemory(device, staging_buffer_memory, in_size_bytes);
        memcpy(mapped_data, in_data, in_size_bytes);
        VulkanUtils::UnmapBufferMemory(device, staging_buffer_memory);
        std::cout << "[Debug PreprocessImage] Data copied to staging buffer." << std::endl;

        // --- 3. Update Persistent Descriptor Set ---
        VkDescriptorImageInfo input_image_info = {};
        input_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;  // Layout for storage image
        input_image_info.imageView = in_image_view;
        input_image_info.sampler = VK_NULL_HANDLE;  // No sampler for storage image

        VkWriteDescriptorSet write_input = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write_input.dstSet = descriptor_set_;
        write_input.dstBinding = 0;  // Binding 0
        write_input.dstArrayElement = 0;
        write_input.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  // Match shader
        write_input.descriptorCount = 1;
        write_input.pImageInfo = &input_image_info;

        vkUpdateDescriptorSets(device, 1, &write_input, 0, nullptr);
        std::cout << "[Debug PreprocessImage] Descriptor set updated." << std::endl;

        // --- 4. Record & Submit Commands ---
        vkResetFences(device, 1, &fence_);
        VkCommandBuffer cmd = context_->BeginOneTimeCommands();

        // a. Copy staging buffer to input image
        VulkanUtils::TransitionImageLayout(cmd, in_image, in_image_format,
                                           VK_IMAGE_LAYOUT_UNDEFINED,
                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VulkanUtils::CopyBufferToImage(cmd, staging_buffer, in_image, in_width, in_height);

        // b. Transition input image for shader read
        VulkanUtils::TransitionImageLayout(cmd, in_image, in_image_format,
                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                           VK_IMAGE_LAYOUT_GENERAL);  // Must be GENERAL

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

        // f. Add barrier to make sure shader writes are finished before copy
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;  // Wait for transfer read
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,  // Wait in transfer stage
                             0, 1, &barrier, 0, nullptr, 0, nullptr);

        // g. Copy device-local buffer to host-visible buffer
        VkBufferCopy copy_region = {};
        copy_region.size = out_size_bytes_;
        vkCmdCopyBuffer(cmd, output_buffer_device_, readback_buffer_, 1, &copy_region);

        // h. Add barrier to make sure copy is finished before host read
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;  // Wait for host read
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,  // Wait in host stage
                             0, 1, &barrier, 0, nullptr, 0, nullptr);

        // i. End command buffer
        vkEndCommandBuffer(cmd);

        // j. Submit commands
        VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;

        std::cout << "[Debug PreprocessImage] Submitting commands..." << std::endl;
        if (vkQueueSubmit(context_->GetComputeQueue(), 1, &submit_info, fence_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit command buffer!");
        }

        // Wait for operations to complete
        vkWaitForFences(device, 1, &fence_, VK_TRUE, UINT64_MAX);
        std::cout << "[Debug PreprocessImage] Commands submitted and awaited." << std::endl;
        vkFreeCommandBuffers(device, context_->GetCommandPool(), 1, &cmd);

        // --- 5. Readback Data ---
        std::cout << "[Debug PreprocessImage] Mapping readback buffer..." << std::endl;
        mapped_data =
            VulkanUtils::MapBufferMemory(device, readback_buffer_memory_, out_size_bytes_);

        std::cout << "[Debug PreprocessImage] Readback buffer mapped. mapped_data=" << mapped_data
                  << ", out_data=" << out_data << std::endl;

        if (out_data == nullptr) {
            throw std::runtime_error("out_data pointer was null before memcpy!");
        }
        if (mapped_data == nullptr) {
            throw std::runtime_error("mapped_data pointer was null before memcpy!");
        }

        std::cout << "[Debug PreprocessImage] Calling memcpy..." << std::endl;
        
        // --- BENCHMARK START ---
        auto start_time = std::chrono::high_resolution_clock::now();
        memcpy(out_data, mapped_data, out_size_bytes_);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "[Debug PreprocessImage] memcpy complete. "
                  << "Time taken: " << duration.count() << " microseconds." << std::endl;
        // --- BENCHMARK END ---

        VulkanUtils::UnmapBufferMemory(device, readback_buffer_memory_);
        std::cout << "[Debug PreprocessImage] Data read back from GPU." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Debug PreprocessImage] EXCEPTION CAUGHT: " << e.what() << std::endl;

        // --- 6. Cleanup (on failure) ---
        std::cout << "[Debug PreprocessImage] Cleaning up resources after exception..."
                  << std::endl;
        if (in_image_view != VK_NULL_HANDLE) vkDestroyImageView(device, in_image_view, nullptr);
        if (in_image != VK_NULL_HANDLE) vkDestroyImage(device, in_image, nullptr);
        if (in_image_memory != VK_NULL_HANDLE) vkFreeMemory(device, in_image_memory, nullptr);
        if (staging_buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, staging_buffer, nullptr);
        if (staging_buffer_memory != VK_NULL_HANDLE)
            vkFreeMemory(device, staging_buffer_memory, nullptr);

        std::cout << "[Debug PreprocessImage] Returning false from catch block." << std::endl;
        return false;
    }

    // --- 6. Cleanup (on success) ---
    std::cout << "[Debug PreprocessImage] Cleaning up resources after success..." << std::endl;
    vkDestroyImageView(device, in_image_view, nullptr);
    vkDestroyImage(device, in_image, nullptr);
    vkFreeMemory(device, in_image_memory, nullptr);
    vkDestroyBuffer(device, staging_buffer, nullptr);
    vkFreeMemory(device, staging_buffer_memory, nullptr);

    std::cout << "[Debug PreprocessImage] Returning true from success path." << std::endl;
    return true;
}

#ifdef __ANDROID__
// --- Main Processing Function (AHardwareBuffer Path) ---
bool VulkanImageProcessor::PreprocessImage(AHardwareBuffer* in_buffer, int in_width, int in_height,
                                           void* out_data) {
    std::cout << "[Debug PreprocessImage-AHB] Entering function." << std::endl;
    if (!context_ || !compute_pipeline_ || !vkGetAndroidHardwareBufferPropertiesANDROID_ ||
        !vkGetMemoryAndroidHardwareBufferANDROID_) {
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

        // We must tell ImportAhbToImage to add the STORAGE_BIT
        if (!VulkanUtils::ImportAhbToImage(
                device, context_->GetPhysicalDevice(), in_buffer,
                vkGetAndroidHardwareBufferPropertiesANDROID_,
                vkGetMemoryAndroidHardwareBufferANDROID_,
                VK_IMAGE_USAGE_STORAGE_BIT,  // <-- Tell it to add this usage
                in_image, in_image_memory, in_image_view, in_image_format)) {
            throw std::runtime_error("Failed to import AHardwareBuffer to VkImage.");
        }
        std::cout << "[Debug PreprocessImage-AHB] AHB imported successfully. Format: "
                  << in_image_format << std::endl;

        // --- 2. Update Persistent Descriptor Set ---
        VkDescriptorImageInfo input_image_info = {};
        input_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;  // Layout for storage
        input_image_info.imageView = in_image_view;
        input_image_info.sampler = VK_NULL_HANDLE;  // No sampler

        VkWriteDescriptorSet write_input = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write_input.dstSet = descriptor_set_;
        write_input.dstBinding = 0;  // Binding 0
        write_input.dstArrayElement = 0;
        write_input.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  // Match shader
        write_input.descriptorCount = 1;
        write_input.pImageInfo = &input_image_info;

        vkUpdateDescriptorSets(device, 1, &write_input, 0, nullptr);
        std::cout << "[Debug PreprocessImage-AHB] Descriptor set updated." << std::endl;

        // --- 3. Record & Submit Commands ---
        vkResetFences(device, 1, &fence_);
        VkCommandBuffer cmd = context_->BeginOneTimeCommands();

        // a. Transition input image layout
        VulkanUtils::TransitionImageLayout(cmd, in_image, in_image_format,
                                           VK_IMAGE_LAYOUT_UNDEFINED,
                                           VK_IMAGE_LAYOUT_GENERAL);  // Must be GENERAL

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

        // e. Add barrier to make sure shader writes are finished before copy
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;  // Wait for transfer read
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,  // Wait in transfer stage
                             0, 1, &barrier, 0, nullptr, 0, nullptr);

        // f. Copy device-local buffer to host-visible buffer
        VkBufferCopy copy_region = {};
        copy_region.size = out_size_bytes_;
        vkCmdCopyBuffer(cmd, output_buffer_device_, readback_buffer_, 1, &copy_region);

        // g. Add barrier to make sure copy is finished before host read
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;  // Wait for host read
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,  // Wait in host stage
                             0, 1, &barrier, 0, nullptr, 0, nullptr);

        // h. End command buffer
        vkEndCommandBuffer(cmd);

        // i. Submit commands
        VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
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
        void* mapped_data =
            VulkanUtils::MapBufferMemory(device, readback_buffer_memory_, out_size_bytes_);

        std::cout << "[Debug PreprocessImage-AHB] Calling memcpy..." << std::endl;
        
        // --- BENCHMARK START ---
        auto start_time_ahb = std::chrono::high_resolution_clock::now();
        memcpy(out_data, mapped_data, out_size_bytes_);
        auto end_time_ahb = std::chrono::high_resolution_clock::now();
        auto duration_ahb = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ahb - start_time_ahb);
        std::cout << "[Debug PreprocessImage-AHB] memcpy complete. "
                  << "Time taken: " << float(duration_ahb.count()) / 1000.0 << " ms. " << std::endl;
        // --- BENCHMARK END ---

        VulkanUtils::UnmapBufferMemory(device, readback_buffer_memory_);
        std::cout << "[Debug PreprocessImage-AHB] Data read back from GPU." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Debug PreprocessImage-AHB] EXCEPTION CAUGHT: " << e.what() << std::endl;
        // --- 5. Cleanup (on failure) ---
        std::cout << "[Debug PreprocessImage-AHB] Cleaning up resources after exception..."
                  << std::endl;
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

bool VulkanImageProcessor::PreprocessImage_ZeroCopy(AHardwareBuffer* in_buffer, int in_width,
                                                    int in_height) {
    std::cerr << "PreprocessImage_ZeroCopy is not implemented." << std::endl;
    return false;
}
#endif  // __ANDROID__

// --- Private Helper Functions Implementation ---

bool VulkanImageProcessor::createPersistentResources() {
    VkDevice device = context_->GetDevice();
    VkPhysicalDevice physical_device = context_->GetPhysicalDevice();

    try {
        // --- MODIFICATION: Create two buffers ---

        // 1. Create the device-local output buffer (written by shader)
        std::cout << "[Debug PreprocessImage] Creating device-local output buffer..." << std::endl;
        VulkanUtils::CreateBuffer(
            device, physical_device, out_size_bytes_,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  // Shader writes, Transfer source
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            output_buffer_device_, output_buffer_device_memory_);

        // 2. Create the host-visible readback buffer (copy destination)
        std::cout << "[Debug PreprocessImage] Creating host-visible readback buffer..." << std::endl;
        VulkanUtils::CreateBuffer(
            device, physical_device, out_size_bytes_,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,  // Transfer destination
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            readback_buffer_, readback_buffer_memory_);

        std::cout << "[Debug PreprocessImage] Output/readback buffers created." << std::endl;
        // --- END MODIFICATION ---

        // 3. Create descriptor pool & set (persistent)
        if (!createDescriptorPool()) {
            throw std::runtime_error("Failed to create descriptor pool.");
        }
        std::cout << "[Debug PreprocessImage] Allocating descriptor set..." << std::endl;
        if (!createDescriptorSet()) {
            throw std::runtime_error("Failed to create descriptor set.");
        }
        std::cout << "[Debug PreprocessImage] Descriptor set allocated and updated." << std::endl;

        // 4. Create fence (persistent)
        VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(device, &fence_info, nullptr, &fence_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fence!");
        }

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
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
        descriptor_set_ = VK_NULL_HANDLE;
    }

    // --- NEW: Clean up device-local buffer ---
    if (output_buffer_device_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, output_buffer_device_, nullptr);
        output_buffer_device_ = VK_NULL_HANDLE;
    }
    if (output_buffer_device_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device, output_buffer_device_memory_, nullptr);
        output_buffer_device_memory_ = VK_NULL_HANDLE;
    }
    // --- END NEW ---

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

    // We need 1 set, with 1 INPUT storage image and 1 OUTPUT storage buffer
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

    // Update Binding 1 (Output Buffer) permanently
    // Binding 0 (Input Image) will be updated in PreprocessImage

    VkDescriptorBufferInfo output_buffer_info = {};
    output_buffer_info.buffer = output_buffer_device_;  // <-- MODIFICATION: Point to device-local buffer
    output_buffer_info.offset = 0;
    output_buffer_info.range = out_size_bytes_;

    VkWriteDescriptorSet write_output = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write_output.dstSet = descriptor_set_;
    write_output.dstBinding = 1;
    write_output.dstArrayElement = 0;
    write_output.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_output.descriptorCount = 1;
    write_output.pBufferInfo = &output_buffer_info;

    vkUpdateDescriptorSets(device, 1, &write_output, 0, nullptr);

    std::cout << "[Debug PreprocessImage] Descriptor set (Binding 1) updated." << std::endl;
    return true;
}