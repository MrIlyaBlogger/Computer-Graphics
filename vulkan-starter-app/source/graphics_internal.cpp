#include "graphics_internal.hpp"
#include "math.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>

#include <fstream>
#include <cstring>

#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>

#include <VkBootstrap.h>
#include <lodepng.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4100 4189 4324)
#endif
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <backends/imgui_impl_vulkan.h>

namespace graphics::internal {

namespace {

VkShaderModule loadShaderModule(const char path[]) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open SPIR-V shader: " << path << '\n';
		return nullptr;
	}
	const std::streamoff file_size = file.tellg();
	if (file_size <= 0 || file_size % sizeof(uint32_t) != 0) {
		std::cerr << "Invalid SPIR-V file size for shader: " << path << '\n';
		return nullptr;
	}
	const size_t size = static_cast<size_t>(file_size);
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	if (!file) {
		std::cerr << "Failed to read SPIR-V shader: " << path << '\n';
		return nullptr;
	}
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};
	VkShaderModule result;
	if (vkCreateShaderModule(context.device, &info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}
	return result;
}

VkInstance vk_instance;
uint32_t vk_api_version;
VkSurfaceKHR vk_surface;

VkSwapchainKHR vk_swapchain;
std::vector<VkImage> vk_swapchain_images;
std::vector<VkImageView> vk_swapchain_image_views;
uint32_t vk_swapchain_current_image;

uint32_t vk_swapchain_resize_width;
uint32_t vk_swapchain_resize_height;
bool vk_swapchain_resize_require;

VkFormat vk_depth_buffer_format = VK_FORMAT_UNDEFINED;
VkImage vk_image_depth_buffer;
VmaAllocation vma_allocation_depth_buffer;
VkImageView vk_image_view_depth_buffer;

std::vector<VkFramebuffer> vk_framebuffers;

VkSemaphore vk_semaphore_image_available;
std::vector<VkSemaphore> vk_semaphores_image_finished;
VkFence vk_fence_frame_in_flight;

VkCommandPool vk_command_pool;
VkCommandBuffer vk_command_buffer;

VkDescriptorPool vk_imgui_descriptor_pool;
VkRenderPass vk_imgui_render_pass;
std::vector<VkFramebuffer> vk_imgui_framebuffers;
VkCommandPool vk_imgui_command_pool;
VkCommandBuffer vk_imgui_command_buffer;

enum TextureIndex {
	WHITE_TEXTURE, BLACK_TEXTURE, CONE_ALBEDO, CONE_ALT_ALBEDO,
	FLOOR_ALBEDO, WALL_ALBEDO, CONE_SPECULAR, CONE_EMISSIVE,
};

template<typename RecordCommands>
bool uploadData(const void* data, VkDeviceSize size, RecordCommands recordCommands) {
	const VkBufferCreateInfo staging_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	};
	const VmaAllocationCreateInfo staging_alloc = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VkBuffer staging{};
	VmaAllocation allocation{};
	VmaAllocationInfo mapped{};
	if (vmaCreateBuffer(context.allocator, &staging_info, &staging_alloc, &staging,
			&allocation, &mapped) != VK_SUCCESS)
		return false;

	const VkCommandBufferAllocateInfo command_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk_command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer command{};
	VkFence fence{};
	const bool uploaded = [&] {
		std::memcpy(mapped.pMappedData, data, static_cast<size_t>(size));
		if (vmaFlushAllocation(context.allocator, allocation, 0, size) != VK_SUCCESS ||
			vkAllocateCommandBuffers(context.device, &command_info, &command) != VK_SUCCESS)
			return false;

		const VkCommandBufferBeginInfo begin = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS)
			return false;
		recordCommands(command, staging);
		if (vkEndCommandBuffer(command) != VK_SUCCESS)
			return false;

		const VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
		const VkSubmitInfo submit = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &command,
		};
		return vkCreateFence(context.device, &fence_info, nullptr, &fence) == VK_SUCCESS &&
			vkQueueSubmit(context.graphics_queue, 1, &submit, fence) == VK_SUCCESS &&
			vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
	}();

	vkDestroyFence(context.device, fence, nullptr);
	if (command)
		vkFreeCommandBuffers(context.device, vk_command_pool, 1, &command);
	vmaDestroyBuffer(context.allocator, staging, allocation);
	return uploaded;
}

const VmaAllocationCreateInfo device_allocation = {
	.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
};

bool uploadGeometry(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
					VkAccessFlags access, VkBuffer& buffer, VmaAllocation& allocation) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &device_allocation, &buffer,
			&allocation, nullptr) != VK_SUCCESS)
		return false;

	if (uploadData(data, size, [&](VkCommandBuffer command, VkBuffer staging) {
		const VkBufferCopy copy = { .size = size };
		vkCmdCopyBuffer(command, staging, buffer, 1, &copy);

		const VkBufferMemoryBarrier barrier = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = access,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.buffer = buffer,
			.size = size,
		};
		vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
	}))
		return true;

	vmaDestroyBuffer(context.allocator, buffer, allocation);
	buffer = VK_NULL_HANDLE;
	allocation = VK_NULL_HANDLE;
	return false;
}

void destroyTextures() {
	for (auto& texture : context.textures) {
		vkDestroyImageView(context.device, texture.view, nullptr);
		vmaDestroyImage(context.allocator, texture.image, texture.allocation);
		texture = {};
	}
	vkDestroySampler(context.device, context.texture_sampler, nullptr);
	context.texture_sampler = VK_NULL_HANDLE;
}

bool uploadTexture(Texture& texture, const unsigned char* pixels, uint32_t width,
				   uint32_t height, VkFormat format) {
	VkFormatProperties format_properties{};
	vkGetPhysicalDeviceFormatProperties(context.physical_device, format, &format_properties);
	constexpr VkFormatFeatureFlags mip_features = VK_FORMAT_FEATURE_BLIT_SRC_BIT |
		VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
	uint32_t mip_levels = 1;
	if ((format_properties.optimalTilingFeatures & mip_features) == mip_features) {
		for (uint32_t size = std::max(width, height); size > 1; size /= 2)
			++mip_levels;
	}

	VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	if (mip_levels > 1)
		usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

	const VkImageCreateInfo image = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = { width, height, 1 },
		.mipLevels = mip_levels,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = usage,
	};
	if (vmaCreateImage(context.allocator, &image, &device_allocation,
			&texture.image, &texture.allocation, nullptr) != VK_SUCCESS)
		return false;

	const VkImageSubresourceRange range = {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.levelCount = mip_levels,
		.layerCount = 1,
	};
	const VkImageViewCreateInfo view = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = texture.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = range,
	};
	if (vkCreateImageView(context.device, &view, nullptr, &texture.view) != VK_SUCCESS)
		return false;

	return uploadData(pixels, VkDeviceSize(width) * height * 4,
		[&](VkCommandBuffer command, VkBuffer staging) {
			VkImageMemoryBarrier barrier = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = texture.image,
				.subresourceRange = range,
			};
			vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

			const VkBufferImageCopy copy = {
				.imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
				.imageExtent = image.extent,
			};
			vkCmdCopyBufferToImage(command, staging, texture.image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

			int32_t mip_width = static_cast<int32_t>(width);
			int32_t mip_height = static_cast<int32_t>(height);
			barrier.subresourceRange.levelCount = 1;
			for (uint32_t level = 1; level < mip_levels; ++level) {
				barrier.subresourceRange.baseMipLevel = level - 1;
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

				const int32_t next_width = std::max(mip_width / 2, 1);
				const int32_t next_height = std::max(mip_height / 2, 1);
				const VkImageBlit blit = {
					.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1 },
					.srcOffsets = { {}, { mip_width, mip_height, 1 } },
					.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1 },
					.dstOffsets = { {}, { next_width, next_height, 1 } },
				};
				vkCmdBlitImage(command, texture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
				mip_width = next_width;
				mip_height = next_height;
			}

			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			if (mip_levels > 1) {
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrier.subresourceRange.baseMipLevel = 0;
				barrier.subresourceRange.levelCount = mip_levels - 1;
				vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
			}

			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.subresourceRange.baseMipLevel = mip_levels - 1;
			barrier.subresourceRange.levelCount = 1;
			vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
		});
}

bool loadTexture(const char* filename, TextureIndex index, VkFormat format, bool& loaded) {
	const std::string local_path = std::string("textures/") + filename;
	const std::string source_path = std::string(LAB_TEXTURE_DIR "/") + filename;
	const std::string& path = std::ifstream(local_path).good() ? local_path : source_path;
	std::vector<unsigned char> pixels;
	unsigned width = 0, height = 0;
	const unsigned error = lodepng::decode(pixels, width, height, path);
	if (error) {
		std::cerr << path << ": " << lodepng_error_text(error) << "; using fallback texture\n";
		return true;
	}
	loaded = uploadTexture(context.textures[index], pixels.data(), width, height, format);
	return loaded;
}

bool createTextures() {
	const unsigned char white[] = { 255, 255, 255, 255 };
	const unsigned char black[] = { 0, 0, 0, 255 };
	const VkFormat color = VK_FORMAT_R8G8B8A8_SRGB;
	const VkFormat mask = VK_FORMAT_R8G8B8A8_UNORM;
	if (!uploadTexture(context.textures[WHITE_TEXTURE], white, 1, 1, color) ||
		!uploadTexture(context.textures[BLACK_TEXTURE], black, 1, 1, mask) ||
		!loadTexture("cone.png", CONE_ALBEDO, color, context.material_has_albedo[CONE_MATERIAL]) ||
		!loadTexture("cone_alt.png", CONE_ALT_ALBEDO, color, context.material_has_albedo[CONE_ALT_MATERIAL]) ||
		!loadTexture("floor.png", FLOOR_ALBEDO, color, context.material_has_albedo[FLOOR_MATERIAL]) ||
		!loadTexture("wall.png", WALL_ALBEDO, color, context.material_has_albedo[WALL_MATERIAL]) ||
		!loadTexture("cone_specular.png", CONE_SPECULAR, mask, context.cone_has_specular) ||
		!loadTexture("cone_emissive.png", CONE_EMISSIVE, color, context.cone_has_emissive)) {
		destroyTextures();
		return false;
	}

	const VkSamplerCreateInfo sampler = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.maxLod = VK_LOD_CLAMP_NONE,
	};
	if (vkCreateSampler(context.device, &sampler, nullptr, &context.texture_sampler) != VK_SUCCESS) {
		destroyTextures();
		return false;
	}
	return true;
}

bool createMaterialSets() {
	const TextureIndex albedo_indices[] = { CONE_ALBEDO, CONE_ALT_ALBEDO, FLOOR_ALBEDO, WALL_ALBEDO };
	const VkDescriptorBufferInfo model_info = {
		.buffer = context.model_uniform_buffer,
		.range = sizeof(ModelUniform),
	};
	const VkDescriptorSetAllocateInfo set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = context.descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &context.model_set_layout,
	};
	for (int i = 0; i < MATERIAL_COUNT; ++i) {
		auto& set = context.material_descriptor_sets[i];
		if (vkAllocateDescriptorSets(context.device, &set_info, &set) != VK_SUCCESS)
			return false;

		const TextureIndex indices[] = {
			context.material_has_albedo[i] ? albedo_indices[i] : WHITE_TEXTURE,
			i == CONE_MATERIAL && context.cone_has_specular ? CONE_SPECULAR : BLACK_TEXTURE,
			i == CONE_MATERIAL && context.cone_has_emissive ? CONE_EMISSIVE : BLACK_TEXTURE,
		};
		VkDescriptorImageInfo images[3]{};
		VkWriteDescriptorSet writes[4] = {{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.pBufferInfo = &model_info,
		}};
		for (uint32_t j = 0; j < 3; ++j) {
			images[j] = { context.texture_sampler, context.textures[indices[j]].view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
			writes[j + 1] = {
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = set,
				.dstBinding = j + 1,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &images[j],
			};
		}
		vkUpdateDescriptorSets(context.device, 4, writes, 0, nullptr);
	}
	return true;
}

VkFormat selectDepthFormat(VkPhysicalDevice physical_device) {
	const VkFormat candidates[] = {
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_D32_SFLOAT_S8_UINT,
	};

	for (VkFormat format : candidates) {
		VkFormatProperties properties{};
		vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
		if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
			return format;
		}
	}

	return VK_FORMAT_UNDEFINED;
}

bool initializeImGUI() {
	const VkDescriptorPoolSize descriptor_pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_SAMPLER,
			.descriptorCount = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLER_POOL_SIZE,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
			.descriptorCount = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE,
		},
	};

	const VkDescriptorPoolCreateInfo descriptor_pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets = uint32_t(vk_swapchain_images.size()),
		.poolSizeCount = sizeof(descriptor_pool_sizes) / sizeof(descriptor_pool_sizes[0]),
		.pPoolSizes = descriptor_pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &descriptor_pool, nullptr,
							   &vk_imgui_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool for ImGUI rendering\n";
		return false;
	}

	const VkAttachmentDescription render_pass_attachment = {
		.format = context.swapchain_format,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
	};

	const VkAttachmentReference render_pass_attachment_ref = {
		.attachment = 0,
		.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	};

	const VkSubpassDescription render_pass_subpass = {
		.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
		.colorAttachmentCount = 1,
		.pColorAttachments = &render_pass_attachment_ref,
	};

	const VkSubpassDependency render_pass_dependency = {
		.srcSubpass = VK_SUBPASS_EXTERNAL,
		.dstSubpass = 0,
		.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
						 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
	};

	const VkRenderPassCreateInfo render_pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &render_pass_attachment,
		.subpassCount = 1,
		.pSubpasses = &render_pass_subpass,
		.dependencyCount = 1,
		.pDependencies = &render_pass_dependency,
	};

	if (vkCreateRenderPass(context.device, &render_pass, nullptr,
						   &vk_imgui_render_pass) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan render pass for ImGUI rendering\n";
		return false;
	}

	const uint32_t swapchain_images_count = uint32_t(vk_swapchain_images.size());

	vk_imgui_framebuffers.resize(swapchain_images_count);

	VkFramebufferCreateInfo framebuffer = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = vk_imgui_render_pass,
		.attachmentCount = 1,
		.width = context.swapchain_extent.width,
		.height = context.swapchain_extent.height,
		.layers = 1,
	};

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		framebuffer.pAttachments = &vk_swapchain_image_views[i];

		if (vkCreateFramebuffer(context.device, &framebuffer, nullptr,
								&vk_imgui_framebuffers[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan framebuffer #" << i << " for ImGUI rendering\n";
			return false;
		}
	}

	const VkCommandPoolCreateInfo command_pool = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = context.graphics_queue_index,
	};

	if (vkCreateCommandPool(context.device, &command_pool, nullptr,
							&vk_imgui_command_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan command pool for ImGUI rendering\n";
		return false;
	}

	const VkCommandBufferAllocateInfo command_buffer = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk_imgui_command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	if (vkAllocateCommandBuffers(context.device, &command_buffer,
								 &vk_imgui_command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan command buffer for ImGUI rendering\n";
		return false;
	}

	ImGui_ImplVulkan_InitInfo init = {
		.ApiVersion = vk_api_version,
		.Instance = vk_instance,
		.PhysicalDevice = context.physical_device,
		.Device = context.device,
		.QueueFamily = context.graphics_queue_index,
		.Queue = context.graphics_queue,
		.DescriptorPool = vk_imgui_descriptor_pool,
		.MinImageCount = swapchain_images_count,
		.ImageCount = swapchain_images_count,
		.PipelineInfoMain = {
			.RenderPass = vk_imgui_render_pass,
		},
	};

	return ImGui_ImplVulkan_Init(&init);
}

void drawImGUI() {
	vkResetCommandBuffer(vk_imgui_command_buffer, 0);

	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(vk_imgui_command_buffer, &command_buffer_begin);

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = vk_imgui_render_pass,
		.framebuffer = vk_imgui_framebuffers[vk_swapchain_current_image],
		.renderArea = { .extent = context.swapchain_extent },
	};

	vkCmdBeginRenderPass(vk_imgui_command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), vk_imgui_command_buffer);

	vkCmdEndRenderPass(vk_imgui_command_buffer);

	vkEndCommandBuffer(vk_imgui_command_buffer);
}

bool rebuildSwapchain(uint32_t width, uint32_t height) {
	if (width == 0 || height == 0) {
		std::cerr << "Cannot rebuild Vulkan swapchain with zero extent\n";
		return false;
	}

	VkSurfaceCapabilitiesKHR surface_capabilities{};
	const VkResult surface_capabilities_result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
		context.physical_device, vk_surface, &surface_capabilities);
	if (surface_capabilities_result != VK_SUCCESS) {
		std::cerr << "Failed to query Vulkan surface capabilities: "
				  << surface_capabilities_result << '\n';
		return false;
	}

	VkExtent2D effective_extent{};
	if (surface_capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		effective_extent = surface_capabilities.currentExtent;
	} else {
		if (surface_capabilities.minImageExtent.width > surface_capabilities.maxImageExtent.width ||
			surface_capabilities.minImageExtent.height > surface_capabilities.maxImageExtent.height) {
			std::cerr << "Invalid Vulkan surface extent limits\n";
			return false;
		}
		effective_extent.width = std::clamp(width, surface_capabilities.minImageExtent.width,
										  surface_capabilities.maxImageExtent.width);
		effective_extent.height = std::clamp(height, surface_capabilities.minImageExtent.height,
									   surface_capabilities.maxImageExtent.height);
	}
	if (effective_extent.width == 0 || effective_extent.height == 0) {
		std::cerr << "Cannot rebuild Vulkan swapchain while surface extent is zero\n";
		return false;
	}

	vkb::SwapchainBuilder sb(context.physical_device, context.device, vk_surface,
	                         context.graphics_queue_index, context.graphics_queue_index);

	auto sb_result = sb.set_desired_extent(width, height)
	                   .use_default_format_selection()
					   .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
					   .use_default_image_usage_flags()
					   .set_old_swapchain(vk_swapchain)
					   .build();
	if (!sb_result) {
		std::cerr << sb_result.error().message() << '\n';
		return false;
	}

	auto vkb_swapchain = sb_result.value();
	if (vkb_swapchain.extent.width == 0 || vkb_swapchain.extent.height == 0) {
		std::cerr << "Swapchain builder returned a zero extent\n";
		return false;
	}

	vkQueueWaitIdle(context.graphics_queue);

	for (size_t i = 0, n = vk_swapchain_image_views.size(); i < n; ++i) {
		vkDestroyFramebuffer(context.device, vk_imgui_framebuffers[i], nullptr);
		vkDestroyFramebuffer(context.device, vk_framebuffers[i], nullptr);
		vkDestroyImageView(context.device, vk_swapchain_image_views[i], nullptr);
	}

	vkDestroySwapchainKHR(context.device, vk_swapchain, nullptr);

	vk_swapchain = vkb_swapchain.swapchain;
	context.swapchain_format = vkb_swapchain.image_format;
	context.swapchain_extent = vkb_swapchain.extent;

	auto swapchain_images = vkb_swapchain.get_images().value();
	auto swapchain_image_views = vkb_swapchain.get_image_views().value();

	vk_swapchain_images = std::move(swapchain_images);
	vk_swapchain_image_views = std::move(swapchain_image_views);

	vkDestroyImageView(context.device, vk_image_view_depth_buffer, nullptr);
	vmaDestroyImage(context.allocator, vk_image_depth_buffer, vma_allocation_depth_buffer);

	const VkImageCreateInfo depth_buffer = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = vk_depth_buffer_format,
		.extent = { context.swapchain_extent.width, context.swapchain_extent.height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const VmaAllocationCreateInfo depth_buffer_allocation = {
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateImage(context.allocator, &depth_buffer, &depth_buffer_allocation,
					   &vk_image_depth_buffer, &vma_allocation_depth_buffer,
					   nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to allocate and create Vulkan image for depth buffer\n";
		return false;
	}

	const VkImageViewCreateInfo depth_buffer_view = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = vk_image_depth_buffer,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = vk_depth_buffer_format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	if (vkCreateImageView(context.device, &depth_buffer_view, nullptr,
						  &vk_image_view_depth_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan image view for depth buffer\n";
		return false;
	}

	const uint32_t swapchain_images_count = uint32_t(vk_swapchain_images.size());

	VkImageView framebuffer_attachments[] = {
		VK_NULL_HANDLE,
		vk_image_view_depth_buffer,
	};

	const VkFramebufferCreateInfo framebuffer = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = context.render_pass,
		.attachmentCount = sizeof(framebuffer_attachments) / sizeof(framebuffer_attachments[0]),
		.pAttachments = framebuffer_attachments,
		.width = context.swapchain_extent.width,
		.height = context.swapchain_extent.height,
		.layers = 1,
	};

	vk_framebuffers.resize(swapchain_images_count);

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		framebuffer_attachments[0] = vk_swapchain_image_views[i];

		if (vkCreateFramebuffer(context.device, &framebuffer, nullptr,
								&vk_framebuffers[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan framebuffer #" << i << '\n';
			return false;
		}
	}

	vk_imgui_framebuffers.resize(swapchain_images_count);

	VkFramebufferCreateInfo imgui_framebuffer = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = vk_imgui_render_pass,
		.attachmentCount = 1,
		.width = context.swapchain_extent.width,
		.height = context.swapchain_extent.height,
		.layers = 1,
	};

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		imgui_framebuffer.pAttachments = &vk_swapchain_image_views[i];

		if (vkCreateFramebuffer(context.device, &imgui_framebuffer, nullptr,
								&vk_imgui_framebuffers[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan framebuffer #" << i << " for ImGUI rendering\n";
			return false;
		}
	}

	vk_swapchain_resize_require = false;

	return true;
}

}

Context context;

bool initialize(GLFWwindow* const window) {

	vkb::InstanceBuilder ib;

	auto ibr = ib.require_api_version(VK_MAKE_VERSION(1, 1, 0))
				 .request_validation_layers()
				 .build();

	if (!ibr) {
		std::cerr << ibr.error().message() << '\n';
		return false;
	}

	auto vkb_instance = ibr.value();
	vk_instance = vkb_instance.instance;
	vk_api_version = vkb_instance.api_version;

	if (glfwCreateWindowSurface(vk_instance, window, nullptr, &vk_surface) != VK_SUCCESS) {
		const char *message = nullptr;
		glfwGetError(&message);
		std::cerr << message << '\n';
		return false;
	}

	vkb::PhysicalDeviceSelector pds(vkb_instance, vk_surface);

	auto pds_result = pds.prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
						 .require_present()
						 .select();
	if (!pds_result) {
		std::cerr << pds_result.error().message() << '\n';
		return false;
	}

	auto vkb_physical_device = pds_result.value();

	vk_depth_buffer_format = selectDepthFormat(vkb_physical_device.physical_device);
	if (vk_depth_buffer_format == VK_FORMAT_UNDEFINED) {
		std::cerr << "No supported depth/stencil attachment format found\n";
		return false;
	}

	vkb::DeviceBuilder db(vkb_physical_device);

	auto db_result = db.build();
	if (!db_result) {
		std::cerr << db_result.error().message() << '\n';
		return false;
	}

	auto vkb_device = db_result.value();

	context.physical_device = vkb_device.physical_device;
	context.device = vkb_device.device;

	if (auto result = vkb_device.get_queue(vkb::QueueType::graphics); result) {
		context.graphics_queue = result.value();
	} else {
		std::cerr << result.error().message() << '\n';
		return false;
	}

	if (auto result = vkb_device.get_queue_index(vkb::QueueType::graphics); result) {
		context.graphics_queue_index = result.value();
	} else {
		std::cerr << result.error().message() << '\n';
		return false;
	}

	const VmaAllocatorCreateInfo allocator = {
		.physicalDevice = context.physical_device,
		.device = context.device,
		.instance = vk_instance,
		.vulkanApiVersion = vk_api_version,
	};

	if (vmaCreateAllocator(&allocator, &context.allocator) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan Memory Allocator\n";
		return false;
	}

	int framebuffer_width = 0;
	int framebuffer_height = 0;
	glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
	if (framebuffer_width <= 0 || framebuffer_height <= 0) {
		std::cerr << "Cannot initialize Vulkan swapchain with zero framebuffer extent\n";
		return false;
	}

	vkb::SwapchainBuilder sb(vkb_device);

	auto sb_result = sb.set_desired_extent(uint32_t(framebuffer_width), uint32_t(framebuffer_height))
					 .use_default_format_selection()
					   .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
					   .use_default_image_usage_flags()
					   .build();
	if (!sb_result) {
		std::cerr << sb_result.error().message() << '\n';
		return false;
	}

	auto vkb_swapchain = sb_result.value();
	if (vkb_swapchain.extent.width == 0 || vkb_swapchain.extent.height == 0) {
		std::cerr << "Swapchain builder returned a zero extent during initialization\n";
		return false;
	}

	vk_swapchain = vkb_swapchain.swapchain;
	context.swapchain_format = vkb_swapchain.image_format;
	context.swapchain_extent = vkb_swapchain.extent;
	vk_swapchain_images = vkb_swapchain.get_images().value();
	vk_swapchain_image_views = vkb_swapchain.get_image_views().value();
	vk_swapchain_current_image = UINT32_MAX;

	const uint32_t swapchain_images_count = uint32_t(vk_swapchain_images.size());

	const VkImageCreateInfo depth_buffer = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = vk_depth_buffer_format,
		.extent = { context.swapchain_extent.width, context.swapchain_extent.height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const VmaAllocationCreateInfo depth_buffer_allocation = {
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateImage(context.allocator, &depth_buffer, &depth_buffer_allocation,
					   &vk_image_depth_buffer, &vma_allocation_depth_buffer,
					   nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to allocate and create Vulkan image for depth buffer\n";
		return false;
	}

	const VkImageViewCreateInfo depth_buffer_view = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = vk_image_depth_buffer,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = vk_depth_buffer_format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	if (vkCreateImageView(context.device, &depth_buffer_view, nullptr,
						  &vk_image_view_depth_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan image view for depth buffer\n";
		return false;
	}

	const VkAttachmentDescription render_pass_attachments[] = {
		{
			.format = context.swapchain_format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		},
		{
			.format = vk_depth_buffer_format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		},
	};

	const VkAttachmentReference render_pass_color_attachment = {
		.attachment = 0,
		.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	};

	const VkAttachmentReference render_pass_depth_attachment = {
		.attachment = 1,
		.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	};

	const VkSubpassDescription render_pass_subpass = {
		.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
		.colorAttachmentCount = 1,
		.pColorAttachments = &render_pass_color_attachment,
		.pDepthStencilAttachment = &render_pass_depth_attachment,
	};

	const VkRenderPassCreateInfo render_pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.attachmentCount = sizeof(render_pass_attachments) / sizeof(render_pass_attachments[0]),
		.pAttachments = render_pass_attachments,
		.subpassCount = 1,
		.pSubpasses = &render_pass_subpass,
	};

	if (vkCreateRenderPass(context.device, &render_pass, nullptr, &context.render_pass) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan render pass\n";
		return false;
	}

	const VkCommandPoolCreateInfo command_pool = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = context.graphics_queue_index,
	};

	if (vkCreateCommandPool(context.device, &command_pool, nullptr, &vk_command_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan command pool\n";
		return false;
	}

	struct Vertex {
		float position[3];
		float uv[2];
		float normal[3];
	};

	constexpr int CONE_SEGMENTS = 50;
	constexpr float PI_F = 3.14159265358979f;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	const auto addVertex = [&vertices, &indices](float x, float y, float z,
										 float nx, float ny, float nz, float u, float v) {
		vertices.push_back({ {x, y, z}, {u, v}, {nx, ny, nz} });
		indices.push_back(static_cast<uint32_t>(vertices.size() - 1));
	};

	for (int i = 0; i < CONE_SEGMENTS; ++i) {
		const float a0 = 2.0f * PI_F * static_cast<float>(i) / CONE_SEGMENTS;
		const float a1 = 2.0f * PI_F * static_cast<float>((i + 1) % CONE_SEGMENTS) / CONE_SEGMENTS;
		const float x0 = std::cos(a0), z0 = std::sin(a0);
		const float x1 = std::cos(a1), z1 = std::sin(a1);
		constexpr float texture_inset = 0.002f;
		constexpr float uv_span = 1.0f - 2.0f * texture_inset;
		constexpr float tip_fraction = 0.14f;
		const float normal_scale = 1.0f / std::sqrt(1.25f);

		const auto addSideVertex = [&](float x, float z, float height_fraction) {
			addVertex(x * height_fraction, 1.0f - 2.0f * height_fraction, z * height_fraction,
				x * normal_scale, 0.5f * normal_scale, z * normal_scale,
				0.5f, texture_inset + height_fraction * uv_span);
		};

		addVertex(0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.5f, texture_inset);
		addSideVertex(x1, z1, tip_fraction);
		addSideVertex(x0, z0, tip_fraction);

		addSideVertex(x0, z0, tip_fraction);
		addSideVertex(x1, z1, tip_fraction);
		addSideVertex(x1, z1, 1.0f);
		addSideVertex(x0, z0, tip_fraction);
		addSideVertex(x1, z1, 1.0f);
		addSideVertex(x0, z0, 1.0f);

		constexpr float bottom_v = 1.0f - texture_inset;
		addVertex(0.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.5f, 0.5f);
		addVertex(x0, -1.0f, z0, 0.0f, -1.0f, 0.0f, 0.5f, bottom_v);
		addVertex(x1, -1.0f, z1, 0.0f, -1.0f, 0.0f, 0.5f, bottom_v);
	}

	context.index_count = static_cast<uint32_t>(indices.size());
	context.sphere_index_start = context.index_count;
	constexpr int SPHERE_RINGS = 8;
	constexpr int SPHERE_SEGMENTS = 12;
	for (int ring = 0; ring < SPHERE_RINGS; ++ring) {
		const float v0 = PI_F * static_cast<float>(ring) / SPHERE_RINGS;
		const float v1 = PI_F * static_cast<float>(ring + 1) / SPHERE_RINGS;
		for (int segment = 0; segment < SPHERE_SEGMENTS; ++segment) {
			const float u0 = 2.0f * PI_F * static_cast<float>(segment) / SPHERE_SEGMENTS;
			const float u1 = 2.0f * PI_F * static_cast<float>(segment + 1) / SPHERE_SEGMENTS;
			const float p[4][3] = {
				{ std::sin(v0) * std::cos(u0), std::cos(v0), std::sin(v0) * std::sin(u0) },
				{ std::sin(v0) * std::cos(u1), std::cos(v0), std::sin(v0) * std::sin(u1) },
				{ std::sin(v1) * std::cos(u1), std::cos(v1), std::sin(v1) * std::sin(u1) },
				{ std::sin(v1) * std::cos(u0), std::cos(v1), std::sin(v1) * std::sin(u0) },
			};
			const float uv[4][2] = {
				{ u0 / (2.0f * PI_F), v0 / PI_F }, { u1 / (2.0f * PI_F), v0 / PI_F },
				{ u1 / (2.0f * PI_F), v1 / PI_F }, { u0 / (2.0f * PI_F), v1 / PI_F },
			};
			const int triangles[2][3] = { { 0, 1, 2 }, { 0, 2, 3 } };
			for (const auto& triangle : triangles) {
				for (int vertex : triangle) {
					addVertex(p[vertex][0], p[vertex][1], p[vertex][2],
						p[vertex][0], p[vertex][1], p[vertex][2], uv[vertex][0], uv[vertex][1]);
				}
			}
		}
	}
	context.sphere_index_count = static_cast<uint32_t>(indices.size()) - context.sphere_index_start;

	constexpr float room_texture_repeats = 4.0f;
	const float room_corners[4][4] = {
		{ -1.0f, -1.0f, 0.0f, room_texture_repeats }, { -1.0f, 1.0f, 0.0f, 0.0f },
		{  1.0f,  1.0f, room_texture_repeats, 0.0f }, { 1.0f, -1.0f, room_texture_repeats, room_texture_repeats },
	};
	const auto addRoomQuad = [&](bool swap_uv, bool flip_v) {
		const auto first_index = static_cast<uint32_t>(indices.size());
		for (int corner : { 0, 1, 2, 0, 2, 3 }) {
			const auto& p = room_corners[corner];
			const float u = swap_uv ? p[3] : p[2];
			float v = swap_uv ? p[2] : p[3];
			if (flip_v)
				v = room_texture_repeats - v;
			addVertex(p[0], 0.0f, p[1], 0.0f, 1.0f, 0.0f, u, v);
		}
		return first_index;
	};
	context.floor_index_start = addRoomQuad(false, false);
	context.back_wall_index_start = addRoomQuad(false, true);
	context.side_wall_index_start = addRoomQuad(true, false);

	if (!uploadGeometry(vertices.data(), vertices.size() * sizeof(Vertex),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			context.vertex_buffer, context.vertex_buffer_allocation) ||
		!uploadGeometry(indices.data(), indices.size() * sizeof(uint32_t),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_ACCESS_INDEX_READ_BIT,
			context.index_buffer, context.index_buffer_allocation) ||
		!createTextures()) {
		std::cerr << "Failed to upload scene geometry or textures\n";
		return false;
	}

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};
	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 3,
		.pBindings = scene_bindings,
	};
	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
									&context.scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout for scene uniforms and lights\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding model_bindings[] = {
		{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
		  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
	};
	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 4,
		.pBindings = model_bindings,
	};
	if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr,
									&context.model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout for model uniforms\n";
		return false;
	}

	const VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 },
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = MATERIAL_COUNT },
		{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 3 * MATERIAL_COUNT },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + MATERIAL_COUNT,
		.poolSizeCount = 4,
		.pPoolSizes = pool_sizes,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &context.descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool\n";
		return false;
	}

	VkBufferCreateInfo ubo_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(SceneUniforms),
		.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VmaAllocationCreateInfo ubo_alloc = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
				 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	vmaCreateBuffer(context.allocator, &ubo_info, &ubo_alloc,
		&context.scene_uniform_buffer,
		&context.scene_uniform_buffer_allocation, nullptr);
	vmaMapMemory(context.allocator, context.scene_uniform_buffer_allocation,
		(void**)&context.scene_uniform_buffer_mapped);

	ubo_info.size = sizeof(PointLight) * Context::LIGHT_BUFFER_CAPACITY;
	ubo_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	if (vmaCreateBuffer(context.allocator, &ubo_info, &ubo_alloc,
		&context.point_light_buffer, &context.point_light_buffer_allocation, nullptr) != VK_SUCCESS ||
		vmaMapMemory(context.allocator, context.point_light_buffer_allocation,
			(void**)&context.point_light_buffer_mapped) != VK_SUCCESS) {
		std::cerr << "Failed to create point light storage buffer\n";
		return false;
	}
	ubo_info.size = sizeof(SpotLight) * Context::LIGHT_BUFFER_CAPACITY;
	if (vmaCreateBuffer(context.allocator, &ubo_info, &ubo_alloc,
		&context.spot_light_buffer, &context.spot_light_buffer_allocation, nullptr) != VK_SUCCESS ||
		vmaMapMemory(context.allocator, context.spot_light_buffer_allocation,
			(void**)&context.spot_light_buffer_mapped) != VK_SUCCESS) {
		std::cerr << "Failed to create spot light storage buffer\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo scene_set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = context.descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &context.scene_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &scene_set_info,
								 &context.scene_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor set for scene data\n";
		return false;
	}

	const VkDescriptorBufferInfo scene_buf_info = {
		.buffer = context.scene_uniform_buffer,
		.offset = 0,
		.range = sizeof(SceneUniforms),
	};
	const VkDescriptorBufferInfo point_buf_info = {
		.buffer = context.point_light_buffer,
		.offset = 0,
		.range = sizeof(PointLight) * Context::LIGHT_BUFFER_CAPACITY,
	};
	const VkDescriptorBufferInfo spot_buf_info = {
		.buffer = context.spot_light_buffer,
		.offset = 0,
		.range = sizeof(SpotLight) * Context::LIGHT_BUFFER_CAPACITY,
	};
	const VkWriteDescriptorSet scene_writes[] = {
		{
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = context.scene_descriptor_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &scene_buf_info,
		},
		{
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = context.scene_descriptor_set,
		.dstBinding = 1,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &point_buf_info,
		},
		{
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = context.scene_descriptor_set,
		.dstBinding = 2,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &spot_buf_info,
		},
	};

	vkUpdateDescriptorSets(context.device, 3, scene_writes, 0, nullptr);

	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(context.physical_device, &properties);
	const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
	context.model_uniform_stride = static_cast<uint32_t>(
		(sizeof(ModelUniform) + alignment - 1) / alignment * alignment);

	ubo_info.size = VkDeviceSize(context.model_uniform_stride) * Context::OBJECT_COUNT;
	ubo_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	VmaAllocationInfo model_allocation{};
	if (vmaCreateBuffer(context.allocator, &ubo_info, &ubo_alloc, &context.model_uniform_buffer,
			&context.model_uniform_buffer_allocation, &model_allocation) != VK_SUCCESS) {
		std::cerr << "Failed to create dynamic model uniform buffer\n";
		return false;
	}
	auto* model_data = static_cast<char*>(model_allocation.pMappedData);
	for (int i = 0; i < Context::OBJECT_COUNT; ++i) {
		context.model_uniform_buffer_mapped[i] =
			reinterpret_cast<ModelUniform*>(model_data + i * context.model_uniform_stride);
	}

	if (!createMaterialSets()) {
		std::cerr << "Failed to create texture material descriptor sets\n";
		return false;
	}

	VkShaderModule vert_shader = loadShaderModule("shaders/shader.vert.spv");
	VkShaderModule frag_shader = loadShaderModule("shaders/shader.frag.spv");
	if (vert_shader == VK_NULL_HANDLE || frag_shader == VK_NULL_HANDLE) {
		std::cerr << "Failed to load vertex/fragment shader modules; check SPIR-V files and working directory\n";
		return false;
	}

	VkPipelineShaderStageCreateInfo shader_stages[2] = {
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert_shader, .pName = "main" },
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag_shader, .pName = "main" },
	};

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position)},
		{.location = 1, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv)},
		{.location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal)},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 3,
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,

	};

	const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkPipelineRasterizationStateCreateInfo rasterizer = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDescriptorSetLayout set_layouts[] = {
	context.scene_set_layout,
	context.model_set_layout,
	};

	const VkPushConstantRange texture_effect_range = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(TextureEffect),
	};

	const VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &texture_effect_range,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr,
			&context.pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		return false;
	}

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input_info,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending,
		.pDynamicState = &dynamic_state,
		.layout = context.pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &context.graphics_pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	vkDestroyShaderModule(context.device, vert_shader, nullptr);
	vkDestroyShaderModule(context.device, frag_shader, nullptr);

	VkImageView framebuffer_attachments[] = {
		VK_NULL_HANDLE,
		vk_image_view_depth_buffer
	};

	const VkFramebufferCreateInfo framebuffer = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = context.render_pass,
		.attachmentCount = sizeof(framebuffer_attachments) / sizeof(framebuffer_attachments[0]),
		.pAttachments = framebuffer_attachments,
		.width = context.swapchain_extent.width,
		.height = context.swapchain_extent.height,
		.layers = 1,
	};

	vk_framebuffers.resize(swapchain_images_count);

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		framebuffer_attachments[0] = vk_swapchain_image_views[i];

		if (vkCreateFramebuffer(context.device, &framebuffer, nullptr,
								&vk_framebuffers[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan framebuffer #" << i << '\n';
			return false;
		}
	}

	const VkSemaphoreCreateInfo semaphore = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

	const VkFenceCreateInfo fence = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT,
	};

	vk_semaphores_image_finished.resize(swapchain_images_count);

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		if (vkCreateSemaphore(context.device, &semaphore, nullptr,
							  &vk_semaphores_image_finished[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan semaphore #" << i << " for finished image\n";
			return false;
		}
	}

	if (vkCreateSemaphore(context.device, &semaphore, NULL,
						  &vk_semaphore_image_available) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan semaphore for available image\n";
		return false;
	}

	if (vkCreateFence(context.device, &fence, nullptr, &vk_fence_frame_in_flight) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan semaphore for image in flight\n";
		return false;
	}

	const VkCommandBufferAllocateInfo command_buffers = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk_command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	if (vkAllocateCommandBuffers(context.device, &command_buffers, &vk_command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan command buffer\n";
		return false;
	}

	if (!initializeImGUI()) {
		std::cerr << "Failed to initialize ImGUI Vulkan rendering backend\n";
		return false;
	}

	return true;
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	ImGui_ImplVulkan_Shutdown();

	vkDestroyPipeline(context.device, context.graphics_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, context.pipeline_layout, nullptr);

	vmaUnmapMemory(context.allocator, context.scene_uniform_buffer_allocation);
	vmaDestroyBuffer(context.allocator, context.scene_uniform_buffer,
		context.scene_uniform_buffer_allocation);
	vmaUnmapMemory(context.allocator, context.point_light_buffer_allocation);
	vmaDestroyBuffer(context.allocator, context.point_light_buffer,
		context.point_light_buffer_allocation);
	vmaUnmapMemory(context.allocator, context.spot_light_buffer_allocation);
	vmaDestroyBuffer(context.allocator, context.spot_light_buffer,
		context.spot_light_buffer_allocation);

	vkDestroyDescriptorPool(context.device, context.descriptor_pool, nullptr);
	vmaDestroyBuffer(context.allocator, context.model_uniform_buffer,
		context.model_uniform_buffer_allocation);
	destroyTextures();

	vkDestroyDescriptorSetLayout(context.device, context.scene_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, context.model_set_layout, nullptr);

	vmaDestroyBuffer(context.allocator, context.vertex_buffer, context.vertex_buffer_allocation);
	vmaDestroyBuffer(context.allocator, context.index_buffer, context.index_buffer_allocation);

	vkDestroyCommandPool(context.device, vk_imgui_command_pool, nullptr);
	for (size_t i = 0, n = vk_imgui_framebuffers.size(); i < n; ++i) {
		vkDestroyFramebuffer(context.device, vk_imgui_framebuffers[i], nullptr);
	}
	vkDestroyRenderPass(context.device, vk_imgui_render_pass, nullptr);
	vkDestroyDescriptorPool(context.device, vk_imgui_descriptor_pool, nullptr);

	vkDestroyCommandPool(context.device, vk_command_pool, nullptr);

	vkDestroyFence(context.device, vk_fence_frame_in_flight, nullptr);
	for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
		vkDestroySemaphore(context.device, vk_semaphores_image_finished[i], nullptr);
	}
	vkDestroySemaphore(context.device, vk_semaphore_image_available, nullptr);

	for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
		vkDestroyFramebuffer(context.device, vk_framebuffers[i], nullptr);
	}
	vkDestroyRenderPass(context.device, context.render_pass, nullptr);

	vkDestroyImageView(context.device, vk_image_view_depth_buffer, nullptr);
	vmaDestroyImage(context.allocator, vk_image_depth_buffer, vma_allocation_depth_buffer);

	for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
		vkDestroyImageView(context.device, vk_swapchain_image_views[i], nullptr);
	}
	vkDestroySwapchainKHR(context.device, vk_swapchain, nullptr);

	vmaDestroyAllocator(context.allocator);

	vkDestroyDevice(context.device, nullptr);

	vkDestroySurfaceKHR(vk_instance, vk_surface, nullptr);
	vkDestroyInstance(vk_instance, nullptr);
}

void resize(uint32_t width, uint32_t height) {
	if (width == 0 || height == 0) {
		return;
	}

	vk_swapchain_resize_width = width;
	vk_swapchain_resize_height = height;

	vk_swapchain_resize_require = true;
}

FrameData prepare() {
	vkWaitForFences(context.device, 1, &vk_fence_frame_in_flight, VK_TRUE, UINT64_MAX);

retry_acquire:
	switch (vkAcquireNextImageKHR(context.device, vk_swapchain, UINT64_MAX,
								  vk_semaphore_image_available, VK_NULL_HANDLE,
								  &vk_swapchain_current_image)) {
	case VK_SUCCESS:
		break;

	case VK_ERROR_OUT_OF_DATE_KHR:
		if (!rebuildSwapchain(vk_swapchain_resize_width, vk_swapchain_resize_height)) {
			return {};
		}
		goto retry_acquire;

	case VK_SUBOPTIMAL_KHR:
		std::cerr << "Swapchain is suboptimal for rendering!\n";
		break;

	default:
		std::cerr << "Failed to acquire Vulkan swapchain image\n";
		return {};
	}

	if (vk_swapchain_current_image >= vk_framebuffers.size() ||
		vk_framebuffers[vk_swapchain_current_image] == VK_NULL_HANDLE) {
		std::cerr << "Acquired swapchain image has no valid framebuffer\n";
		return {};
	}

	vkResetFences(context.device, 1, &vk_fence_frame_in_flight);

	return {
		.framebuffer = vk_framebuffers[vk_swapchain_current_image],
		.command_buffer = vk_command_buffer,
	};
}

void submitAndPresent() {
	drawImGUI();

	const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	const VkCommandBuffer command_buffers[] = {
		vk_command_buffer,
		vk_imgui_command_buffer,
	};

	const VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &vk_semaphore_image_available,
		.pWaitDstStageMask = &stage,
		.commandBufferCount = sizeof(command_buffers) / sizeof(command_buffers[0]),
		.pCommandBuffers = command_buffers,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &vk_semaphores_image_finished[vk_swapchain_current_image],
	};

	vkQueueSubmit(context.graphics_queue, 1, &submit, vk_fence_frame_in_flight);

	const VkPresentInfoKHR present = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &vk_semaphores_image_finished[vk_swapchain_current_image],
		.swapchainCount = 1,
		.pSwapchains = &vk_swapchain,
		.pImageIndices = &vk_swapchain_current_image,
	};

	VkResult result = vkQueuePresentKHR(context.graphics_queue, &present);
	if (result == VK_ERROR_OUT_OF_DATE_KHR ||
	    result == VK_SUBOPTIMAL_KHR ||
	    vk_swapchain_resize_require) {
		rebuildSwapchain(vk_swapchain_resize_width, vk_swapchain_resize_height);
	} else if (result != VK_SUCCESS) {
		std::cerr << "Failed to present Vulkan swapchain image\n";
	}
}

}
