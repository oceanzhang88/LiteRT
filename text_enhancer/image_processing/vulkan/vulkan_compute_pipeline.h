#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

#include "vulkan_context.h"

// Define the push constant structure matching the shader
struct CropResizePushConstants {
    int32_t in_dims[2];
    int32_t crop_dims[2];
    int32_t out_dims[2];
};

class VulkanComputePipeline {
   public:
    VulkanComputePipeline();
    ~VulkanComputePipeline();

    // Initializes shader, layouts, and pipeline
    bool Initialize(VulkanContext* context, const std::string& shader_spirv_path);
    // Destroys all pipeline-related objects
    void Shutdown();

    // Getters
    VkPipeline GetPipeline() const { return compute_pipeline_; }
    VkPipelineLayout GetPipelineLayout() const { return pipeline_layout_; }
    VkDescriptorSetLayout GetDescriptorSetLayout() const { return descriptor_set_layout_; }

   private:
    VkShaderModule createShaderModule(const std::vector<char>& code);

    VulkanContext* context_ = nullptr;  // Non-owning pointer
    VkDevice device_ = VK_NULL_HANDLE;  // Cached for convenience

    // Compute pipeline resources
    VkPipeline compute_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkShaderModule compute_shader_module_ = VK_NULL_HANDLE;
};