#include "vulkan_compute_pipeline.h"

#include <iostream>
#include <stdexcept>

#include "vulkan_utils.h"  // For LoadShaderSPIRV

VulkanComputePipeline::VulkanComputePipeline() {
    // All members initialized to VK_NULL_HANDLE or nullptr in header
}

VulkanComputePipeline::~VulkanComputePipeline() { Shutdown(); }

bool VulkanComputePipeline::Initialize(VulkanContext* context,
                                       const std::string& shader_spirv_path) {
    if (!context) {
        return false;
    }
    context_ = context;
    device_ = context->GetDevice();  // Cache the device handle

    try {
        // 1. Load SPIR-V
        std::vector<char> shader_code = VulkanUtils::LoadShaderSPIRV(shader_spirv_path);
        compute_shader_module_ = createShaderModule(shader_code);
        if (compute_shader_module_ == VK_NULL_HANDLE) {
            throw std::runtime_error("Failed to create shader module.");
        }

        // 2. Create Descriptor Set Layout

        // --- FIX (Binding 0): Input Storage Image ---
        VkDescriptorSetLayoutBinding input_storage_binding = {};
        input_storage_binding.binding = 0;
        input_storage_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        input_storage_binding.descriptorCount = 1;
        input_storage_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // --- OPTIMIZATION (Binding 1): Output Storage Buffer ---
        VkDescriptorSetLayoutBinding output_storage_binding = {};
        output_storage_binding.binding = 1;
        output_storage_binding.descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;  // <-- THE CHANGE
        output_storage_binding.descriptorCount = 1;
        output_storage_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        // --- END OPTIMIZATION ---

        std::vector<VkDescriptorSetLayoutBinding> bindings = {input_storage_binding,
                                                              output_storage_binding};

        VkDescriptorSetLayoutCreateInfo layout_info = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_set_layout_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout!");
        }

        // 3. Create Pipeline Layout (with push constant)
        VkPushConstantRange push_constant_range = {};
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = sizeof(CropResizePushConstants);

        VkPipelineLayoutCreateInfo pipeline_layout_info = {
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant_range;

        if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }

        // 4. Create Compute Pipeline
        VkPipelineShaderStageCreateInfo shader_stage_info = {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shader_stage_info.module = compute_shader_module_;
        shader_stage_info.pName = "main";

        VkComputePipelineCreateInfo pipeline_info = {
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = shader_stage_info;
        pipeline_info.layout = pipeline_layout_;

        if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                     &compute_pipeline_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute pipeline!");
        }

    } catch (const std::exception& e) {
        std::cerr << "Vulkan Compute Pipeline Initialization Error: " << e.what() << std::endl;
        Shutdown();  // Clean up partial initialization
        return false;
    }

    std::cout << "Vulkan compute pipeline created." << std::endl;
    return true;
}

void VulkanComputePipeline::Shutdown() {
    if (device_ == VK_NULL_HANDLE) return;

    if (compute_pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, compute_pipeline_, nullptr);
        compute_pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
        descriptor_set_layout_ = VK_NULL_HANDLE;
    }
    if (compute_shader_module_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, compute_shader_module_, nullptr);
        compute_shader_module_ = VK_NULL_HANDLE;
    }

    device_ = VK_NULL_HANDLE;
    context_ = nullptr;
}

VkShaderModule VulkanComputePipeline::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo create_info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    create_info.codeSize = code.size();
    create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }
    return shader_module;
}