#pragma once

#include "graphics_internal.hpp"

namespace application {

bool initialize();
void shutdown();

void update(double time, GLFWwindow* window);
void render(const graphics::internal::FrameData& fd);

} // namespace application
