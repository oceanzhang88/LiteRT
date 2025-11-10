#pragma once

#include <vector>
#include <string>
#include <vulkan/vulkan.h>

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#include <vulkan/vulkan_android.h> // For PFN types
#endif


namespace VulkanUtils {

// --- Shader/Memory Helpers ---
std::vector<char> LoadShaderSPIRV(const std::string& filepath);

uint32_t FindMemoryType(VkPhysicalDevice physical_device,
                        uint32_t type_filter,
                        VkMemoryPropertyFlags properties);

// --- Resource Creation ---
bool CreateBuffer(VkDevice device,
                  VkPhysicalDevice physical_device,
                  VkDeviceSize size,
                  VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags properties,
                  VkBuffer& buffer,
                  VkDeviceMemory& buffer_memory);

bool CreateImage(VkDevice device,
                 VkPhysicalDevice physical_device,
                 uint32_t width,
                 uint32_t height,
                 VkFormat format,
                 VkImageTiling tiling,
                 VkImageUsageFlags usage,
                 VkMemoryPropertyFlags properties,
                 VkImage& image,
                 VkDeviceMemory& image_memory);

VkImageView CreateImageView(VkDevice device,
                            VkImage image,
                            VkFormat format);

VkSampler CreateSampler(VkDevice device);

#ifdef __ANDROID__
/**
 * @brief Imports an AHardwareBuffer into a new VkImage.
 * @return true on success, false on failure.
 */
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
                      VkFormat& out_format);
#endif // __ANDROID__


// --- Command Helpers ---
void TransitionImageLayout(VkCommandBuffer command_buffer,
                           VkImage image,
                           VkFormat format,
                           VkImageLayout old_layout,
                           VkImageLayout new_layout);

void CopyBufferToImage(VkCommandBuffer command_buffer,
                       VkBuffer buffer,
                       VkImage image,
                       uint32_t width,
                       uint32_t height);

void CopyImageToBuffer(VkCommandBuffer command_buffer,
                       VkImage image,
                       VkBuffer buffer,
                       uint32_t width,
                       uint32_t height);

// --- Memory Mapping ---
void* MapBufferMemory(VkDevice device,
                      VkDeviceMemory buffer_memory,
                      VkDeviceSize size);

void UnmapBufferMemory(VkDevice device,
                       VkDeviceMemory buffer_memory);

} // namespace VulkanUtils