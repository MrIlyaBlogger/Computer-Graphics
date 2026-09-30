#include "application.hpp"
#include "math.hpp"

#include <imgui.h>
#include <cstring>
#include <cmath>
#include <iostream>

namespace {

    struct ObjectState {
        float position[3] = { 0.0f, 0.0f, 0.0f };
        float rotation[3] = { 0.0f, 0.0f, 0.0f };
        float scale[3] = { 1.0f, 1.0f, 1.0f };
        float color[3] = { 1.0f, 1.0f, 1.0f };
        float orbit_radius = 0.0f;
        float orbit_height = 0.0f;
    };

    ObjectState objects[graphics::internal::Context::OBJECT_COUNT] = {
        { {0,0,0}, {0,0,0}, {1,1,1}, {1,1,1}, 2.0f, 0.5f },
    };

    bool  use_perspective = true;
    bool  animate = true;
    float anim_speed = 4.0f;
    float camera_distance = 6.0f;
    float anim_time = 0.0f;
    double last_time = 0.0;

} // namespace

namespace application {

    bool initialize() { return true; }

    void shutdown() {
        auto& context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);
    }

    void update(double time) {
        auto& ctx = graphics::internal::context;

        float dt = static_cast<float>(time - last_time);
        last_time = time;
        if (animate) {
            anim_time += dt * anim_speed;
        }

        float aspect = static_cast<float>(ctx.swapchain_extent.width) /
            static_cast<float>(ctx.swapchain_extent.height);

        math::Mat4 projection{};
        if (use_perspective) {
            math::perspective(1.2f, aspect, 0.1f, 100.0f, projection);
            projection[1][1] = -projection[1][1];
        }
        else {
            float h = 2.5f;
            math::ortho(-h * aspect, h * aspect, -h, h, 0.1f, 100.0f, projection);
            projection[1][1] = -projection[1][1];
        }

        math::Mat4 view{};
        math::translate(0.0f, 0.0f, -camera_distance, view);

        std::memcpy(ctx.scene_uniform_buffer_mapped->projection, projection, sizeof(projection));
        std::memcpy(ctx.scene_uniform_buffer_mapped->view, view, sizeof(view));
        if (vmaFlushAllocation(ctx.allocator, ctx.scene_uniform_buffer_allocation,
                               0, VK_WHOLE_SIZE) != VK_SUCCESS) {
            std::cerr << "Failed to flush scene uniform buffer memory\n";
            return;
        }

        for (int i = 0; i < graphics::internal::Context::OBJECT_COUNT; ++i) {
            auto& s = objects[i];

            float ox = s.position[0] + std::cos(anim_time) * s.orbit_radius;
            float oy = s.position[1] + std::sin(anim_time * 1.7f) * s.orbit_height;
            float oz = s.position[2] + std::sin(anim_time) * s.orbit_radius;

            math::Mat4 mT{}, mRx{}, mRy{}, mRz{}, mS{}, tmp{};
            math::translate(ox, oy, oz, mT);
            math::rotateX(s.rotation[0] * 3.14159f / 180.0f + anim_time, mRx);
            math::rotateY(s.rotation[1] * 3.14159f / 180.0f + anim_time, mRy);
            math::rotateZ(s.rotation[2] * 3.14159f / 180.0f + anim_time, mRz);
            math::scale(s.scale[0], s.scale[1], s.scale[2], mS);

            math::multiply(mT, mRz, tmp);
            math::multiply(tmp, mRy, tmp);
            math::multiply(tmp, mRx, tmp);
            math::multiply(tmp, mS, tmp);

            auto* u = ctx.model_uniform_buffer_mapped[i];
            std::memcpy(u->model, tmp, sizeof(tmp));
            u->color[0] = s.color[0];
            u->color[1] = s.color[1];
            u->color[2] = s.color[2];
            u->color[3] = 1.0f;
            if (vmaFlushAllocation(ctx.allocator, ctx.model_uniform_buffer_allocations[i],
                                   0, VK_WHOLE_SIZE) != VK_SUCCESS) {
                std::cerr << "Failed to flush model uniform buffer memory\n";
                return;
            }
        }

        ImGui::Begin("Lab #1: Cone");

        ImGui::Text("Projection");
        if (ImGui::RadioButton("Perspective", use_perspective)) use_perspective = true;
        ImGui::SameLine();
        if (ImGui::RadioButton("Orthographic", !use_perspective)) use_perspective = false;
        ImGui::SliderFloat("Camera distance", &camera_distance, 1.0f, 15.0f);

        ImGui::Separator();
        ImGui::Text("Animation");
        ImGui::Checkbox("Play", &animate);
        ImGui::SliderFloat("Speed", &anim_speed, 0.0f, 5.0f);

        for (int i = 0; i < graphics::internal::Context::OBJECT_COUNT; ++i) {
            auto& s = objects[i];
            ImGui::PushID(i);
            ImGui::Separator();
            ImGui::Text("Object %d", i + 1);
            ImGui::SliderFloat3("Position", s.position, -3.0f, 3.0f);
            ImGui::SliderFloat3("Rotation (deg)", s.rotation, -180.0f, 180.0f);
            ImGui::SliderFloat3("Scale", s.scale, 0.1f, 3.0f);
            ImGui::ColorEdit3("Color", s.color);
            ImGui::SliderFloat("Orbit radius", &s.orbit_radius, 0.0f, 3.0f);
            ImGui::SliderFloat("Orbit height", &s.orbit_height, -1.0f, 1.0f);
            ImGui::PopID();
        }

        ImGui::End();
    }

    void render(const graphics::internal::FrameData& fd) {
        auto& ctx = graphics::internal::context;

        const VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(fd.command_buffer, &begin_info);

        const VkClearValue clear_values[] = {
            {.color = {.float32 = { 0.05f, 0.05f, 0.08f, 1.0f } } },
            {.depthStencil = { 1.0f, 0 } },
        };
        const VkRenderPassBeginInfo rp = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = ctx.render_pass,
            .framebuffer = fd.framebuffer,
            .renderArea = {.extent = ctx.swapchain_extent },
            .clearValueCount = 2,
            .pClearValues = clear_values,
        };
        vkCmdBeginRenderPass(fd.command_buffer, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx.graphics_pipeline);

        const VkViewport viewport = {
            .x = 0, .y = 0,
            .width = static_cast<float>(ctx.swapchain_extent.width),
            .height = static_cast<float>(ctx.swapchain_extent.height),
            .minDepth = 0, .maxDepth = 1,
        };
        const VkRect2D scissor = { .extent = ctx.swapchain_extent };
        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        const VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &ctx.vertex_buffer, offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, ctx.index_buffer, 0, VK_INDEX_TYPE_UINT32);

        for (int i = 0; i < graphics::internal::Context::OBJECT_COUNT; ++i) {
            const VkDescriptorSet sets[] = {
                ctx.scene_descriptor_set,
                ctx.model_descriptor_sets[i],
            };
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                ctx.pipeline_layout,
                0, 2, sets, 0, nullptr);
            vkCmdDrawIndexed(fd.command_buffer, ctx.index_count, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application