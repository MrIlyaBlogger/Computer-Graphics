#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_normal;

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_world_position;
layout(location = 2) out vec3 out_normal;

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
  out_color = in_color;
}
