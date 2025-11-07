#include "vulkan_utils.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#include <vulkan/vulkan_android.h>
#endif

namespace VulkanUtils {

std::vector<char> LoadShaderSPIRV(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filepath);
    }
    size_t file_size = (size_t)file.tellg();
    std::vector<char> buffer(file_size);
    file.seekg(0);
    file.read(buffer.data(), file_size);
    file.close();
    return buffer;
}

uint32_t FindMemoryType(VkPhysicalDevice physical_device,
                        uint32_t type_filter,
                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type!");
}

bool CreateBuffer(VkDevice device,
                  VkPhysicalDevice physical_device,
                  VkDeviceSize size,
                  VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags properties,
                  VkBuffer& buffer,
                  VkDeviceMemory& buffer_memory) {
    VkBufferCreateInfo buffer_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer!");
    }

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

    VkMemoryAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = FindMemoryType(physical_device, mem_requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, buffer_memory, 0);
    return true;
}

bool CreateImage(VkDevice device,
                 VkPhysicalDevice physical_device,
                 uint32_t width,
                 uint32_t height,
                 VkFormat format,
                 VkImageTiling tiling,
                 VkImageUsageFlags usage,
                 VkMemoryPropertyFlags properties,
                 VkImage& image,
                 VkDeviceMemory& image_memory) {
    VkImageCreateInfo image_info = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = width;
    image_info.extent.height = height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = format;
    image_info.tiling = tiling;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = usage;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image!");
    }

    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(device, image, &mem_requirements);

    VkMemoryAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = FindMemoryType(physical_device, mem_requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &alloc_info, nullptr, &image_memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, image_memory, 0);
    return true;
}

VkImageView CreateImageView(VkDevice device, VkImage image, VkFormat format) {
    VkImageViewCreateInfo view_info = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    VkImageView image_view;
    if (vkCreateImageView(device, &view_info, nullptr, &image_view) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view!");
    }
    return image_view;
}

VkSampler CreateSampler(VkDevice device) {
    VkSamplerCreateInfo sampler_info = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    VkSampler sampler;
    if (vkCreateSampler(device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create texture sampler!");
    }
    return sampler;
}

#ifdef __ANDROID__
// --- FIX: Updated signature to 10 arguments ---
bool ImportAhbToImage(VkDevice device,
                      VkPhysicalDevice physical_device,
                      AHardwareBuffer* hardware_buffer,
                      PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAndroidHardwareBufferPropertiesANDROID,
                      PFN_vkGetMemoryAndroidHardwareBufferANDROID vkGetMemoryAndroidHardwareBufferANDROID,
                      VkImageUsageFlags extra_usage_flags,
                      VkImage& out_image,
                      VkDeviceMemory& out_memory,
                      VkImageView& out_image_view,
                      VkFormat& out_format) {
    
    // (Note: vkGetMemoryAndroidHardwareBufferANDROID is passed but not used by this import path)

    VkAndroidHardwareBufferPropertiesANDROID ahb_props = { VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID };
    VkAndroidHardwareBufferFormatPropertiesANDROID ahb_format_props = { VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID };
    ahb_props.pNext = &ahb_format_props;

    if (vkGetAndroidHardwareBufferPropertiesANDROID(device, hardware_buffer, &ahb_props) != VK_SUCCESS) {
        std::cerr << "Failed to get AHB properties." << std::endl;
        return false;
    }
    
    AHardwareBuffer_Desc ahb_desc;
    AHardwareBuffer_describe(hardware_buffer, &ahb_desc);

    uint32_t memory_type_index = FindMemoryType(physical_device,
                                                ahb_props.memoryTypeBits,
                                                0); // AHB memory is device-local

    VkExternalMemoryImageCreateInfo external_mem_info = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
    external_mem_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    VkImageCreateInfo image_info = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    image_info.pNext = &external_mem_info;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = ahb_desc.width;
    image_info.extent.height = ahb_desc.height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = ahb_format_props.format;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    
    // --- FIX: Use the extra_usage_flags ---
    // We need to be able to read this as a storage image
    image_info.usage = extra_usage_flags | VK_IMAGE_USAGE_SAMPLED_BIT; // Add base usage
    // --- END FIX ---
    
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(device, &image_info, nullptr, &out_image) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create external image for AHB!");
    }

    VkImportAndroidHardwareBufferInfoANDROID import_mem_info = { VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID };
    import_mem_info.buffer = hardware_buffer;

    VkMemoryAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    alloc_info.pNext = &import_mem_info;
    alloc_info.allocationSize = ahb_props.allocationSize;
    alloc_info.memoryTypeIndex = memory_type_index;

    if (vkAllocateMemory(device, &alloc_info, nullptr, &out_memory) != VK_SUCCESS) {
        vkDestroyImage(device, out_image, nullptr);
        throw std::runtime_error("Failed to allocate/import memory for AHB!");
    }

    if (vkBindImageMemory(device, out_image, out_memory, 0) != VK_SUCCESS) {
        vkDestroyImage(device, out_image, nullptr);
        vkFreeMemory(device, out_memory, nullptr);
        throw std::runtime_error("Failed to bind AHB memory to image!");
    }

    out_format = ahb_format_props.format;
    out_image_view = CreateImageView(device, out_image, out_format);

    return true;
}
#endif // __ANDROID__

void TransitionImageLayout(VkCommandBuffer command_buffer,
                           VkImage image,
                           VkFormat format,
                           VkImageLayout old_layout,
                           VkImageLayout new_layout) {
    VkImageMemoryBarrier barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags source_stage;
    VkPipelineStageFlags destination_stage;

    // --- FIX: Added transitions to/from GENERAL layout ---
    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destination_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_GENERAL) {
        // Transition for output storage image
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destination_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_GENERAL && new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        // Transition for reading back output image
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        source_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        // Transition for AHB imported as sampler
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destination_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_GENERAL) {
        // Transition for input storage image (after copy)
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destination_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_GENERAL) {
        // Transition for input storage image (AHB)
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destination_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    // --- END FIX ---
    else {
        // Keep the original logic as a fallback, but log it
        std::cerr << "Unsupported layout transition from " << old_layout << " to " << new_layout << std::endl;
        throw std::invalid_argument("Unsupported layout transition!");
    }


    vkCmdPipelineBarrier(
        command_buffer,
        source_stage, destination_stage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}

void CopyBufferToImage(VkCommandBuffer command_buffer,
                       VkBuffer buffer,
                       VkImage image,
                       uint32_t width,
                       uint32_t height) {
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { width, height, 1 };

    vkCmdCopyBufferToImage(command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void CopyImageToBuffer(VkCommandBuffer command_buffer,
                       VkImage image,
                       VkBuffer buffer,
                       uint32_t width,
                       uint32_t height) {
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { width, height, 1 };

    vkCmdCopyImageToBuffer(command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
}

void* MapBufferMemory(VkDevice device,
                      VkDeviceMemory buffer_memory,
                      VkDeviceSize size) {
    void* mapped_memory;
    std::cout << "[Debug mapBufferMemory] Calling vkMapMemory... (device=" << (void*)device 
              << ", memory=" << (void*)buffer_memory << ", size=" << size << ")" << std::endl;
              
    VkResult result = vkMapMemory(device, buffer_memory, 0, size, 0, &mapped_memory);
    
    std::cout << "[Debug mapBufferMemory] vkMapMemory returned. Result: " << result << std::endl;

    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to map buffer memory!");
    }
    
    std::cout << "[Debug mapBufferMemory] Returning mapped pointer: " << mapped_memory << std::endl;
    return mapped_memory;
}

void UnmapBufferMemory(VkDevice device,
                       VkDeviceMemory buffer_memory) {
    std::cout << "[Debug unmapBufferMemory] Calling vkUnmapMemory..." << std::endl;
    vkUnmapMemory(device, buffer_memory);
    std::cout << "[Debug unmapBufferMemory] vkUnmapMemory returned." << std::endl;
}

} // namespace VulkanUtils