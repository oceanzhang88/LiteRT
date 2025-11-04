#include "vulkan_context.h"
#include <iostream>
#include <stdexcept>
#include <vector>

// --- ADD THIS BLOCK ---
#ifdef __ANDROID__
#include <vulkan/vulkan_android.h> // For AHB extension names
#endif
// --- END OF BLOCK ---


// --- ADD THIS BLOCK ---

// For debug builds, we enable validation layers
#ifdef NDEBUG
const bool kEnableValidationLayers = false;
#else
const bool kEnableValidationLayers = true;
#endif

// Names of the validation layers we want to enable
const std::vector<const char*> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// --- END OF BLOCK ---
// ... (Debug callback helpers are unchanged)
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
    }
    return VK_FALSE;
}
static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}
static void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}
// ... (rest of the file is unchanged up to BeginOneTimeCommands)

VulkanContext::VulkanContext() { }
VulkanContext::~VulkanContext() { Shutdown(); }
bool VulkanContext::Initialize() {
    try {
        if (!createInstance()) {
            std::cerr << "Failed to create Vulkan instance." << std::endl;
            return false;
        }
        std::cout << "Vulkan instance created." << std::endl;

        if (kEnableValidationLayers && !setupDebugMessenger()) {
            std::cout << "Warning: Failed to set up debug messenger." << std::endl;
        }

        if (!findPhysicalDevice()) {
            std::cerr << "Failed to find a suitable physical device." << std::endl;
            return false;
        }
        std::cout << "Vulkan physical device found." << std::endl;

        if (!createDevice()) {
            std::cerr << "Failed to create logical device." << std::endl;
            return false;
        }
        std::cout << "Vulkan logical device created." << std::endl;

        if (!createCommandPool()) {
            std::cerr << "Failed to create command pool." << std::endl;
            return false;
        }
        std::cout << "Vulkan command pool created." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Vulkan Context Initialization Error: " << e.what() << std::endl;
        Shutdown(); // Clean up partial initialization
        return false;
    }
    std::cout << "VulkanContext initialized successfully." << std::endl;
    return true;
}
void VulkanContext::Shutdown() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }

    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
    }

    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    if (kEnableValidationLayers && debug_messenger_ != VK_NULL_HANDLE) {
        DestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
        debug_messenger_ = VK_NULL_HANDLE;
    }

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
    
    physical_device_ = VK_NULL_HANDLE;
    compute_queue_ = VK_NULL_HANDLE;
}
bool VulkanContext::createInstance() {
    VkApplicationInfo app_info = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app_info.pApplicationName = "ImageProcessor";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo create_info = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    create_info.pApplicationInfo = &app_info;

    std::vector<const char*> extensions = {};
    if (kEnableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    // --- ADD THIS BLOCK ---
    #ifdef __ANDROID__
    // Required for AHB support
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    #endif
    // --- END OF BLOCK ---
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    if (kEnableValidationLayers) {
        create_info.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
        create_info.ppEnabledLayerNames = kValidationLayers.data();
    } else {
        create_info.enabledLayerCount = 0;
    }

    return vkCreateInstance(&create_info, nullptr, &instance_) == VK_SUCCESS;
}
bool VulkanContext::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT create_info = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
    create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = debugCallback;
    create_info.pUserData = nullptr;

    return CreateDebugUtilsMessengerEXT(instance_, &create_info, nullptr, &debug_messenger_) == VK_SUCCESS;
}
bool VulkanContext::findPhysicalDevice() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0) {
        return false;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    for (const auto& device : devices) {
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

        for (uint32_t i = 0; i < queue_families.size(); ++i) {
            if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physical_device_ = device;
                compute_queue_family_index_ = i;
                return true;
            }
        }
    }
    return false;
}
bool VulkanContext::createDevice() {
    VkDeviceQueueCreateInfo queue_create_info = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queue_create_info.queueFamilyIndex = compute_queue_family_index_;
    queue_create_info.queueCount = 1;
    float queue_priority = 1.0f;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures device_features = {}; // No special features needed

    VkDeviceCreateInfo create_info = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    create_info.pQueueCreateInfos = &queue_create_info;
    create_info.queueCreateInfoCount = 1;
    create_info.pEnabledFeatures = &device_features;
    
    // --- MODIFY THIS BLOCK ---
    #ifdef __ANDROID__
    // Required device extensions for AHB
    std::vector<const char*> device_extensions = {
        VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME
    };
    create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.data();
    #else
    create_info.enabledExtensionCount = 0;
    #endif
    // --- END OF BLOCK ---

    if (kEnableValidationLayers) {
        create_info.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
        create_info.ppEnabledLayerNames = kValidationLayers.data();
    } else {
        create_info.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) != VK_SUCCESS) {
        return false;
    }

    vkGetDeviceQueue(device_, compute_queue_family_index_, 0, &compute_queue_);
    return true;
}
bool VulkanContext::createCommandPool() {
    VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pool_info.queueFamilyIndex = compute_queue_family_index_;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; 

    return vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) == VK_SUCCESS;
}

VkCommandBuffer VulkanContext::BeginOneTimeCommands() {
    VkCommandBufferAllocateInfo alloc_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool_;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);

    VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(command_buffer, &begin_info);
    
    // --- LOG RESTORED ---
    std::cout << "[Debug PreprocessImage] Beginning one-time commands..." << std::endl;
    // --- END LOG ---

    return command_buffer;
}

void VulkanContext::EndAndSubmitCommands(VkCommandBuffer command_buffer) {
    vkEndCommandBuffer(command_buffer);

    VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    VkFence fence;
    VkFenceCreateInfo fence_info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(device_, &fence_info, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence!");
    }

    // --- LOG RESTORED ---
    std::cout << "[Debug PreprocessImage] Submitting commands..." << std::endl;
    // --- END LOG ---
    if (vkQueueSubmit(compute_queue_, 1, &submit_info, fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer!");
    }

    vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
    
    // --- LOG RESTORED ---
    std::cout << "[Debug PreprocessImage] Commands submitted and awaited." << std::endl;
    // --- END LOG ---
    
    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
}