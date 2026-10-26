#version 450 core

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_world_position;
layout(location = 2) in vec3 in_normal;
layout(location = 0) out vec4 out_color;

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

struct PointLight {
  vec4 position_intensity;
  vec4 color;
};

struct SpotLight {
  vec4 position_intensity;
  vec4 direction_inner_cos;
  vec4 color_outer_cos;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
  PointLight lights[];
}
point_lights;

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
  SpotLight lights[];
}
spot_lights;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
  mat4 model;
  vec4 albedo;
  vec4 specular_shininess;
}
model_uniforms;

vec3 blinnPhong(vec3 normal, vec3 light_direction, vec3 view_direction,
                vec3 light_color, float intensity) {
  float diffuse_factor = max(dot(normal, light_direction), 0.0);
  vec3 half_direction = normalize(light_direction + view_direction);
  float specular_factor = 0.0;
  if (diffuse_factor > 0.0) {
    specular_factor = pow(max(dot(normal, half_direction), 0.0),
                          model_uniforms.specular_shininess.w);
  }
  vec3 diffuse = model_uniforms.albedo.rgb * diffuse_factor;
  vec3 specular = model_uniforms.specular_shininess.rgb * specular_factor;
  return (diffuse + specular) * light_color * intensity;
}

void main() {
  if (model_uniforms.specular_shininess.w <= 0.0) {
    out_color = vec4(model_uniforms.albedo.rgb, 1.0);
    return;
  }

  vec3 normal = normalize(in_normal);
  vec3 view_direction = normalize(scene_uniforms.camera_position.xyz - in_world_position);
  vec3 result = model_uniforms.albedo.rgb * scene_uniforms.ambient_color.rgb;

  vec3 directional_light = normalize(-scene_uniforms.directional_direction.xyz);
  result += blinnPhong(normal, directional_light, view_direction,
                       scene_uniforms.directional_color.rgb,
                       scene_uniforms.directional_color.w);

  int point_count = int(scene_uniforms.light_counts.x);
  for (int i = 0; i < point_count; ++i) {
    vec3 delta = point_lights.lights[i].position_intensity.xyz - in_world_position;
    float distance_squared = max(dot(delta, delta), 0.01);
    vec3 light_direction = normalize(delta);
    float attenuation = point_lights.lights[i].position_intensity.w / distance_squared;
    result += blinnPhong(normal, light_direction, view_direction,
                         point_lights.lights[i].color.rgb, attenuation);
  }

  int spot_count = int(scene_uniforms.light_counts.y);
  for (int i = 0; i < spot_count; ++i) {
    SpotLight light = spot_lights.lights[i];
    vec3 delta = in_world_position - light.position_intensity.xyz;
    float distance_squared = max(dot(delta, delta), 0.01);
    vec3 from_light = normalize(delta);
    vec3 light_direction = -from_light;
    float cone_cos = dot(from_light, normalize(light.direction_inner_cos.xyz));
    float cone = smoothstep(light.color_outer_cos.w,
                            light.direction_inner_cos.w, cone_cos);
    float attenuation = light.position_intensity.w * cone / distance_squared;
    result += blinnPhong(normal, light_direction, view_direction,
                         light.color_outer_cos.rgb, attenuation);
  }

  out_color = vec4(clamp(result * in_color, 0.0, 1.0), 1.0);
}
