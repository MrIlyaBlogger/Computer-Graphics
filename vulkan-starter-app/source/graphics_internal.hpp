#pragma once

#include <cstdint>

#include <vulkan/vulkan_core.h>

#include <vk_mem_alloc.h>

struct GLFWwindow;

namespace graphics::internal {

struct SceneUniforms {
    float projection[4][4];
    float view[4][4];
    float camera_position[4];
    float directional_direction[4];
    float directional_color[4];
    float ambient_color[4];
    float light_counts[4];
};

struct ModelUniform {
    float model[4][4];
    float albedo[4];
    float specular_shininess[4];
};

struct PointLight {
    float position_intensity[4];
    float color[4];
};

struct SpotLight {
    float position_intensity[4];
    float direction_inner_cos[4];
    float color_outer_cos[4];
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
    uint32_t sphere_index_start;
    uint32_t sphere_index_count;

    VkDescriptorSetLayout scene_set_layout;
    VkBuffer scene_uniform_buffer;
    VmaAllocation scene_uniform_buffer_allocation;
    SceneUniforms* scene_uniform_buffer_mapped;
    VkDescriptorSet scene_descriptor_set;

    VkBuffer point_light_buffer;
    VmaAllocation point_light_buffer_allocation;
    PointLight* point_light_buffer_mapped;
    VkBuffer spot_light_buffer;
    VmaAllocation spot_light_buffer_allocation;
    SpotLight* spot_light_buffer_mapped;
    static constexpr uint32_t LIGHT_BUFFER_CAPACITY = 32;
    static constexpr int OBJECT_COUNT = 1 + 2 * LIGHT_BUFFER_CAPACITY;
    VkDescriptorSetLayout model_set_layout;
    VkBuffer model_uniform_buffers[OBJECT_COUNT];
    VmaAllocation model_uniform_buffer_allocations[OBJECT_COUNT];
    ModelUniform* model_uniform_buffer_mapped[OBJECT_COUNT];
    VkDescriptorSet model_descriptor_sets[OBJECT_COUNT];

    VkDescriptorPool descriptor_pool;
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
