#pragma once

#include <cstdint>

#include <vulkan/vulkan_core.h>

#include <vk_mem_alloc.h>

struct GLFWwindow;

namespace graphics::internal {

struct GlobalUniforms {
	float matrix[4][4];
};

struct Context {
    VkPhysicalDevice physical_device;
    VkDevice device;

    VmaAllocator allocator;

    VkQueue graphics_queue;
    uint32_t graphics_queue_index;

    VkFormat swapchain_format;
    VkExtent2D swapchain_extent;

    VkRenderPass render_pass;

    VkPipelineLayout pipeline_layout;
    VkPipeline graphics_pipeline;

    VkBuffer vertex_buffer;
    VmaAllocation vertex_buffer_allocation;

    VkBuffer index_buffer;
    VmaAllocation index_buffer_allocation;
    uint32_t index_count;

    VkBuffer uniform_buffer;
    VmaAllocation uniform_buffer_allocation;
    GlobalUniforms* uniform_buffer_mapped;

    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
};

struct FrameData {
	VkFramebuffer framebuffer;
	VkCommandBuffer command_buffer;
};

extern Context context;

bool initialize(GLFWwindow* const window);
void shutdown();

void resize(uint32_t width, uint32_t height);

FrameData prepare();
void submitAndPresent();

} // namespace graphics::internal