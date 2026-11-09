#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_normal;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_world_position;
layout(location = 2) out vec3 out_normal;
layout(location = 3) out vec2 out_local_xz;
layout(location = 4) flat out uint out_bottom_face;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
  mat4 projection;
  mat4 view;
  vec4 camera_position;
  vec4 directional_direction;
  vec4 directional_color;
  vec4 ambient_color;
  vec4 light_counts;
}
scene_uniforms;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
  mat4 model;
  vec4 albedo;
  vec4 specular_shininess;
}
model_uniforms;

void main() {
  vec4 world_position = model_uniforms.model * vec4(in_position, 1.0);
  gl_Position = scene_uniforms.projection * scene_uniforms.view * world_position;
  out_world_position = world_position.xyz;
  out_normal = transpose(inverse(mat3(model_uniforms.model))) * in_normal;
  out_uv = in_uv;
  out_local_xz = in_position.xz;
  out_bottom_face = in_normal.y < 0.0 ? 1u : 0u;
}
