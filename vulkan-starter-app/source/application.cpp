#include "application.hpp"
#include "math.hpp"

#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

    struct ObjectState {
        float position[3] = { 0.0f, 0.0f, 0.0f };
        float rotation[3] = { 0.0f, 0.0f, 0.0f };
        float scale[3] = { 0.7f, 0.7f, 0.7f };
        float albedo[3] = { 0.55f, 0.7f, 1.0f };
        float specular[3] = { 1.0f, 1.0f, 1.0f };
        float shininess = 32.0f;
    };

    struct PointLightState {
        float position[3];
        float intensity;
        float color[3];
    };

    struct SpotLightState {
        float position[3];
        float intensity;
        float direction[3];
        float inner_angle;
        float outer_angle;
        float color[3];
    };

    ObjectState object;
    float camera_position[3] = { 0.0f, 0.0f, 6.0f };
    float camera_yaw = 0.0f;
    float camera_pitch = 0.0f;
    float camera_fov = 68.75f;
    float directional_direction[3] = { -0.4f, -1.0f, -0.3f };
    float directional_color[3] = { 1.0f, 0.95f, 0.85f };
    float directional_intensity = 1.0f;
    float ambient_color[3] = { 0.08f, 0.08f, 0.1f };
    bool show_light_visualizers = true;
    std::vector<PointLightState> point_lights = {
        { { 2.0f, 1.5f, 2.0f }, 8.0f, { 1.0f, 0.45f, 0.2f } },
    };
    std::vector<SpotLightState> spot_lights;

    constexpr float PI_F = 3.14159265358979f;
    constexpr float DEG_TO_RAD = PI_F / 180.0f;

    void flushBuffer(VmaAllocation allocation, const char* message) {
        if (vmaFlushAllocation(graphics::internal::context.allocator, allocation,
                               0, VK_WHOLE_SIZE) != VK_SUCCESS) {
            std::cerr << message << '\n';
        }
    }

} // namespace

namespace application {

    bool initialize() { return true; }

    void shutdown() {
        auto& context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);
    }

    void update(double time) {
        auto& ctx = graphics::internal::context;
        (void)time;

        const float aspect = static_cast<float>(ctx.swapchain_extent.width) /
            static_cast<float>(ctx.swapchain_extent.height);

        math::Mat4 projection{};
        math::perspective(camera_fov * DEG_TO_RAD, aspect, 0.1f, 100.0f, projection);
        projection[1][1] = -projection[1][1];

        const float yaw = camera_yaw * DEG_TO_RAD;
        const float pitch = camera_pitch * DEG_TO_RAD;
        const float direction[3] = {
            std::sin(yaw) * std::cos(pitch),
            std::sin(pitch),
            -std::cos(yaw) * std::cos(pitch)
        };
        const float target[3] = {
            camera_position[0] + direction[0],
            camera_position[1] + direction[1],
            camera_position[2] + direction[2]
        };
        const float world_up[3] = { 0.0f, 1.0f, 0.0f };
        math::Mat4 view{};
        math::lookAt(camera_position, target, world_up, view);

        auto* scene = ctx.scene_uniform_buffer_mapped;
        std::memcpy(scene->projection, projection, sizeof(projection));
        std::memcpy(scene->view, view, sizeof(view));
        scene->camera_position[0] = camera_position[0];
        scene->camera_position[1] = camera_position[1];
        scene->camera_position[2] = camera_position[2];
        scene->camera_position[3] = 1.0f;
        scene->directional_direction[0] = directional_direction[0];
        scene->directional_direction[1] = directional_direction[1];
        scene->directional_direction[2] = directional_direction[2];
        scene->directional_direction[3] = 0.0f;
        scene->directional_color[0] = directional_color[0];
        scene->directional_color[1] = directional_color[1];
        scene->directional_color[2] = directional_color[2];
        scene->directional_color[3] = directional_intensity;
        scene->ambient_color[0] = ambient_color[0];
        scene->ambient_color[1] = ambient_color[1];
        scene->ambient_color[2] = ambient_color[2];
        scene->ambient_color[3] = 1.0f;
        scene->light_counts[0] = static_cast<float>(point_lights.size());
        scene->light_counts[1] = static_cast<float>(spot_lights.size());
        scene->light_counts[2] = scene->light_counts[3] = 0.0f;
        flushBuffer(ctx.scene_uniform_buffer_allocation,
                    "Failed to flush scene uniform buffer memory");

        std::fill_n(ctx.point_light_buffer_mapped, graphics::internal::Context::LIGHT_BUFFER_CAPACITY,
                    graphics::internal::PointLight{});
        for (size_t i = 0; i < point_lights.size(); ++i) {
            const auto& light = point_lights[i];
            auto& gpu_light = ctx.point_light_buffer_mapped[i];
            gpu_light.position_intensity[0] = light.position[0];
            gpu_light.position_intensity[1] = light.position[1];
            gpu_light.position_intensity[2] = light.position[2];
            gpu_light.position_intensity[3] = light.intensity;
            gpu_light.color[0] = light.color[0];
            gpu_light.color[1] = light.color[1];
            gpu_light.color[2] = light.color[2];
            gpu_light.color[3] = 1.0f;
        }
        flushBuffer(ctx.point_light_buffer_allocation,
                    "Failed to flush point light storage buffer memory");

        std::fill_n(ctx.spot_light_buffer_mapped, graphics::internal::Context::LIGHT_BUFFER_CAPACITY,
                    graphics::internal::SpotLight{});
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            const auto& light = spot_lights[i];
            auto& gpu_light = ctx.spot_light_buffer_mapped[i];
            gpu_light.position_intensity[0] = light.position[0];
            gpu_light.position_intensity[1] = light.position[1];
            gpu_light.position_intensity[2] = light.position[2];
            gpu_light.position_intensity[3] = light.intensity;
            gpu_light.direction_inner_cos[0] = light.direction[0];
            gpu_light.direction_inner_cos[1] = light.direction[1];
            gpu_light.direction_inner_cos[2] = light.direction[2];
            gpu_light.direction_inner_cos[3] = std::cos(light.inner_angle * DEG_TO_RAD);
            gpu_light.color_outer_cos[0] = light.color[0];
            gpu_light.color_outer_cos[1] = light.color[1];
            gpu_light.color_outer_cos[2] = light.color[2];
            gpu_light.color_outer_cos[3] = std::cos(light.outer_angle * DEG_TO_RAD);
        }
        flushBuffer(ctx.spot_light_buffer_allocation,
                    "Failed to flush spot light storage buffer memory");

        math::Mat4 mT{}, mRx{}, mRy{}, mRz{}, mS{}, model{};
        math::translate(object.position[0], object.position[1], object.position[2], mT);
        math::rotateX(object.rotation[0] * DEG_TO_RAD, mRx);
        math::rotateY(object.rotation[1] * DEG_TO_RAD, mRy);
        math::rotateZ(object.rotation[2] * DEG_TO_RAD, mRz);
        math::scale(object.scale[0], object.scale[1], object.scale[2], mS);
        math::multiply(mT, mRz, model);
        math::multiply(model, mRy, model);
        math::multiply(model, mRx, model);
        math::multiply(model, mS, model);

        auto* model_uniform = ctx.model_uniform_buffer_mapped[0];
        std::memcpy(model_uniform->model, model, sizeof(model));
        model_uniform->albedo[0] = object.albedo[0];
        model_uniform->albedo[1] = object.albedo[1];
        model_uniform->albedo[2] = object.albedo[2];
        model_uniform->albedo[3] = 1.0f;
        model_uniform->specular_shininess[0] = object.specular[0];
        model_uniform->specular_shininess[1] = object.specular[1];
        model_uniform->specular_shininess[2] = object.specular[2];
        model_uniform->specular_shininess[3] = object.shininess;
        flushBuffer(ctx.model_uniform_buffer_allocations[0],
                    "Failed to flush model uniform buffer memory");

        const auto writeLightMarker = [&ctx](int index, const float position[3],
                                             const float color[3], float radius) {
            math::Mat4 marker_translation{}, marker_scale{}, marker_model{};
            math::translate(position[0], position[1], position[2], marker_translation);
            math::scale(radius, radius, radius, marker_scale);
            math::multiply(marker_translation, marker_scale, marker_model);
            auto* marker = ctx.model_uniform_buffer_mapped[index];
            std::memcpy(marker->model, marker_model, sizeof(marker_model));
            marker->albedo[0] = color[0];
            marker->albedo[1] = color[1];
            marker->albedo[2] = color[2];
            marker->albedo[3] = 1.0f;
            marker->specular_shininess[0] = 0.0f;
            marker->specular_shininess[1] = 0.0f;
            marker->specular_shininess[2] = 0.0f;
            marker->specular_shininess[3] = 0.0f;
            flushBuffer(ctx.model_uniform_buffer_allocations[index],
                        "Failed to flush light visualizer model buffer memory");
        };
        for (size_t i = 0; i < point_lights.size(); ++i) {
            writeLightMarker(1 + static_cast<int>(i), point_lights[i].position,
                             point_lights[i].color, 0.14f);
        }
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            const auto& light = spot_lights[i];
            float direction_length = std::sqrt(light.direction[0] * light.direction[0] +
                light.direction[1] * light.direction[1] + light.direction[2] * light.direction[2]);
            if (direction_length < 0.001f) direction_length = 1.0f;
            const float spot_direction[3] = {
                light.direction[0] / direction_length,
                light.direction[1] / direction_length,
                light.direction[2] / direction_length
            };
            const float axis_y[3] = { -spot_direction[0], -spot_direction[1], -spot_direction[2] };
            const float spot_world_up[3] = { 0.0f, 1.0f, 0.0f };
            float axis_x[3] = {
                axis_y[1] * spot_world_up[2] - axis_y[2] * spot_world_up[1],
                axis_y[2] * spot_world_up[0] - axis_y[0] * spot_world_up[2],
                axis_y[0] * spot_world_up[1] - axis_y[1] * spot_world_up[0]
            };
            float axis_x_length = std::sqrt(axis_x[0] * axis_x[0] + axis_x[1] * axis_x[1] + axis_x[2] * axis_x[2]);
            if (axis_x_length < 0.001f) {
                axis_x[0] = 1.0f; axis_x[1] = 0.0f; axis_x[2] = 0.0f;
                axis_x_length = 1.0f;
            }
            for (float& v : axis_x) v /= axis_x_length;
            const float axis_z[3] = {
                axis_x[1] * axis_y[2] - axis_x[2] * axis_y[1],
                axis_x[2] * axis_y[0] - axis_x[0] * axis_y[2],
                axis_x[0] * axis_y[1] - axis_x[1] * axis_y[0]
            };
            constexpr float beam_length = 1.2f;
            const float beam_radius = std::min(beam_length *
                std::tan(light.outer_angle * DEG_TO_RAD), 2.5f);
            math::Mat4 rotation{}, marker_scale{}, marker_translation{}, marker_model{};
            math::identity(rotation);
            for (int row = 0; row < 3; ++row) {
                rotation[0][row] = axis_x[row];
                rotation[1][row] = axis_y[row];
                rotation[2][row] = axis_z[row];
            }
            math::scale(beam_radius, beam_length * 0.5f, beam_radius, marker_scale);
            const float center[3] = {
                light.position[0] + spot_direction[0] * beam_length * 0.5f,
                light.position[1] + spot_direction[1] * beam_length * 0.5f,
                light.position[2] + spot_direction[2] * beam_length * 0.5f
            };
            math::translate(center[0], center[1], center[2], marker_translation);
            math::multiply(rotation, marker_scale, marker_model);
            math::multiply(marker_translation, marker_model, marker_model);
            const int marker_index = 1 + graphics::internal::Context::LIGHT_BUFFER_CAPACITY +
                static_cast<int>(i);
            auto* marker = ctx.model_uniform_buffer_mapped[marker_index];
            std::memcpy(marker->model, marker_model, sizeof(marker_model));
            marker->albedo[0] = light.color[0];
            marker->albedo[1] = light.color[1];
            marker->albedo[2] = light.color[2];
            marker->albedo[3] = 1.0f;
            marker->specular_shininess[0] = 0.0f;
            marker->specular_shininess[1] = 0.0f;
            marker->specular_shininess[2] = 0.0f;
            marker->specular_shininess[3] = 0.0f;
            flushBuffer(ctx.model_uniform_buffer_allocations[marker_index],
                        "Failed to flush spot light visualizer model buffer memory");
        }

        ImGui::Begin("Lab #2: Camera and lighting");
        ImGui::Checkbox("Show light visualizers", &show_light_visualizers);

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Position", camera_position, -10.0f, 10.0f);
            ImGui::SliderFloat("Yaw (deg)", &camera_yaw, -180.0f, 180.0f);
            ImGui::SliderFloat("Pitch (deg)", &camera_pitch, -89.0f, 89.0f);
            ImGui::SliderFloat("Field of view", &camera_fov, 25.0f, 110.0f);
        }

        if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Direction", directional_direction, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", directional_color);
            ImGui::SliderFloat("Intensity", &directional_intensity, 0.0f, 5.0f);
            ImGui::ColorEdit3("Ambient light", ambient_color);
        }

        if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (point_lights.size() < graphics::internal::Context::LIGHT_BUFFER_CAPACITY &&
                ImGui::Button("Add point light")) {
                point_lights.push_back({ { 0.0f, 2.0f, 2.0f }, 8.0f, { 0.3f, 0.6f, 1.0f } });
            }
            for (size_t i = 0; i < point_lights.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                ImGui::Separator();
                ImGui::Text("Point light %d", static_cast<int>(i + 1));
                ImGui::SliderFloat3("Position", point_lights[i].position, -8.0f, 8.0f);
                ImGui::ColorEdit3("Color", point_lights[i].color);
                ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 40.0f);
                if (ImGui::Button("Remove point light")) {
                    point_lights.erase(point_lights.begin() + i);
                    ImGui::PopID();
                    break;
                }
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("Spot lights")) {
            if (spot_lights.size() < graphics::internal::Context::LIGHT_BUFFER_CAPACITY &&
                ImGui::Button("Add spot light")) {
                spot_lights.push_back({ { 0.0f, 3.0f, 2.0f }, 20.0f,
                    { 0.0f, -1.0f, -0.5f }, 18.0f, 28.0f, { 1.0f, 0.9f, 0.55f } });
            }
            for (size_t i = 0; i < spot_lights.size(); ++i) {
                ImGui::PushID(static_cast<int>(1000 + i));
                ImGui::Separator();
                ImGui::Text("Spot light %d", static_cast<int>(i + 1));
                ImGui::SliderFloat3("Position", spot_lights[i].position, -8.0f, 8.0f);
                ImGui::SliderFloat3("Direction", spot_lights[i].direction, -1.0f, 1.0f);
                ImGui::ColorEdit3("Color", spot_lights[i].color);
                ImGui::SliderFloat("Intensity", &spot_lights[i].intensity, 0.0f, 80.0f);
                ImGui::SliderFloat("Inner angle", &spot_lights[i].inner_angle, 1.0f, 85.0f);
                ImGui::SliderFloat("Outer angle", &spot_lights[i].outer_angle,
                                   spot_lights[i].inner_angle, 89.0f);
                if (ImGui::Button("Remove spot light")) {
                    spot_lights.erase(spot_lights.begin() + i);
                    ImGui::PopID();
                    break;
                }
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("Object material", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Position", object.position, -3.0f, 3.0f);
            ImGui::SliderFloat3("Rotation (deg)", object.rotation, -180.0f, 180.0f);
            ImGui::SliderFloat3("Scale", object.scale, 0.1f, 3.0f);
            ImGui::ColorEdit3("Albedo", object.albedo);
            ImGui::ColorEdit3("Specular color", object.specular);
            ImGui::SliderFloat("Shininess", &object.shininess, 1.0f, 128.0f);
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

        const int draw_count = 1 + (show_light_visualizers
            ? static_cast<int>(point_lights.size() + spot_lights.size()) : 0);
        for (int i = 0; i < draw_count; ++i) {
            int model_index = i;
            if (i > 1 + static_cast<int>(point_lights.size()) - 1) {
                model_index = 1 + graphics::internal::Context::LIGHT_BUFFER_CAPACITY +
                    i - 1 - static_cast<int>(point_lights.size());
            }
            const VkDescriptorSet sets[] = {
                ctx.scene_descriptor_set,
                ctx.model_descriptor_sets[model_index],
            };
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                ctx.pipeline_layout, 0, 2, sets, 0, nullptr);
            if (i > 0 && i <= static_cast<int>(point_lights.size())) {
                vkCmdDrawIndexed(fd.command_buffer, ctx.sphere_index_count, 1,
                                 ctx.sphere_index_start, 0, 0);
            }
            else {
                vkCmdDrawIndexed(fd.command_buffer, ctx.index_count, 1, 0, 0, 0);
            }
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application
