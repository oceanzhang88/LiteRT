#pragma once

#include <vulkan/vulkan.h>

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    // Initializes instance, device, queue, and command pool
    bool Initialize();
    // Destroys all created Vulkan objects
    void Shutdown();

    // Getters for other modules to use
    VkInstance GetInstance() const { return instance_; }
    VkDevice GetDevice() const { return device_; }
    VkPhysicalDevice GetPhysicalDevice() const { return physical_device_; }
    VkQueue GetComputeQueue() const { return compute_queue_; }
    VkCommandPool GetCommandPool() const { return command_pool_; }
    uint32_t GetComputeQueueFamilyIndex() const { return compute_queue_family_index_; }

    // --- NEW: Getters for timestamp queries ---
    VkQueryPool GetQueryPool() const { return query_pool_; }
    float GetTimestampPeriod() const { return timestamp_period_; }
    // --- END NEW ---

    // Command buffer helpers
    VkCommandBuffer BeginOneTimeCommands();
    void EndAndSubmitCommands(VkCommandBuffer command_buffer);

private:
    bool createInstance();
    bool setupDebugMessenger();
    bool findPhysicalDevice();
    bool createDevice();
    bool createCommandPool();

    // Core Vulkan objects
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t compute_queue_family_index_ = 0;

    // Resource management
    VkCommandPool command_pool_ = VK_NULL_HANDLE;

    // --- NEW: Timestamp query resources ---
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    float timestamp_period_ = 1.0f; // Nanoseconds per tick
    // --- END NEW ---
};