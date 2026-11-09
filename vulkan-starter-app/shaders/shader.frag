#version 450 core

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_world_position;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec2 in_local_xz;
layout(location = 4) flat in uint in_bottom_face;
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

layout(set = 1, binding = 1) uniform sampler2D albedo_texture;
layout(set = 1, binding = 2) uniform sampler2D specular_texture;
layout(set = 1, binding = 3) uniform sampler2D emissive_texture;

layout(push_constant) uniform TextureEffect {
  float time;
  uint flags;
} texture_effect;

const uint ANIMATED_SAMPLING = 1u << 0;
const uint CONE_MAPPING = 1u << 1;
const float PI = 3.14159265;

vec3 sampleMaterialTexture(sampler2D map_texture, vec2 uv, vec2 center_uv, float angular_weight) {
  if ((texture_effect.flags & CONE_MAPPING) == 0u)
    return texture(map_texture, uv).rgb;

  uv.y = clamp(uv.y, 0.002, 0.998);
  center_uv.y = clamp(center_uv.y, 0.002, 0.998);
  vec2 dx = dFdx(uv);
  vec2 dy = dFdy(uv);
  vec2 center_dx = dFdx(center_uv);
  vec2 center_dy = dFdy(center_uv);

  dx.x -= 2.0 * round(dx.x * 0.5);
  dy.x -= 2.0 * round(dy.x * 0.5);
  uv.x = fract(uv.x);

  const float seam_width = 0.05;
  float seam_distance = uv.x < 0.5 ? uv.x : uv.x - 1.0;
  vec3 color;
  if (abs(seam_distance) < seam_width) {
    vec2 left_uv = vec2(min(1.0 + seam_distance, 0.998), uv.y);
    vec2 right_uv = vec2(max(seam_distance, 0.002), uv.y);
    vec3 left_color = textureGrad(map_texture, left_uv, dx, dy).rgb;
    vec3 right_color = textureGrad(map_texture, right_uv, dx, dy).rgb;
    color = mix(left_color, right_color, smoothstep(-seam_width, seam_width, seam_distance));
  } else {
    color = textureGrad(map_texture, uv, dx, dy).rgb;
  }

  if (angular_weight < 1.0) {
    vec3 center_color = textureGrad(map_texture, center_uv, center_dx, center_dy).rgb;
    color = mix(center_color, color, angular_weight);
  }
  return color;
}

vec2 animatedUV(vec2 uv) {
  float t = texture_effect.time;
  vec2 wave = vec2(sin(18.0 * uv.y + 2.1 * t),
                   sin(6.2831853 * 3.0 * uv.x - 1.7 * t));
  return uv + 0.035 * wave;
}

vec3 sampleAlbedo(vec2 uv, vec2 center_uv, float angular_weight) {
  if ((texture_effect.flags & ANIMATED_SAMPLING) == 0u)
    return sampleMaterialTexture(albedo_texture, uv, center_uv, angular_weight);

  float t = texture_effect.time;
  vec2 offset = (0.012 + 0.007 * sin(t * 2.0)) * vec2(cos(t), sin(t));
  vec3 channels = vec3(
      sampleMaterialTexture(albedo_texture, uv + offset, center_uv + offset, angular_weight).r,
      sampleMaterialTexture(albedo_texture, uv, center_uv, angular_weight).g,
      sampleMaterialTexture(albedo_texture, uv - offset, center_uv - offset, angular_weight).b);
  vec3 tint = 0.82 + 0.18 * cos(vec3(0.0, 2.0944, 4.1888) +
                               1.7 * t + 14.0 * mix(center_uv.y, uv.y, angular_weight));
  return channels * tint;
}

vec3 blinnPhong(vec3 albedo, vec3 specular_color,
                vec3 normal, vec3 light_direction, vec3 view_direction,
                vec3 light_color, float intensity) {
  float diffuse_factor = max(dot(normal, light_direction), 0.0);
  vec3 half_direction = normalize(light_direction + view_direction);
  float specular_factor = 0.0;
  if (diffuse_factor > 0.0) {
    specular_factor = pow(max(dot(normal, half_direction), 0.0),
                          model_uniforms.specular_shininess.w);
  }
  vec3 diffuse = albedo * diffuse_factor;
  vec3 specular = specular_color * specular_factor;
  return (diffuse + specular) * light_color * intensity;
}

void main() {
  if (model_uniforms.specular_shininess.w <= 0.0) {
    out_color = vec4(model_uniforms.albedo.rgb, 1.0);
    return;
  }

  vec2 uv = in_uv;
  vec2 center_uv = vec2(0.5, uv.y);
  float angular_weight = 1.0;
  if ((texture_effect.flags & CONE_MAPPING) != 0u) {
    if (dot(in_local_xz, in_local_xz) > 1e-10)
      uv.x = abs(in_local_xz.x) > 1e-6
          ? 0.5 + atan(in_local_xz.y, -in_local_xz.x) / PI
          : 0.5 + 0.5 * sign(in_local_xz.y);
    angular_weight = in_bottom_face == 0u
        ? smoothstep(0.002, 0.13, uv.y)
        : smoothstep(0.50, 0.65, uv.y);
  }

  if ((texture_effect.flags & ANIMATED_SAMPLING) != 0u) {
    uv = animatedUV(uv);
    center_uv = animatedUV(center_uv);
  }
  vec3 albedo = sampleAlbedo(uv, center_uv, angular_weight);
  vec3 specular_color = model_uniforms.specular_shininess.rgb *
                        sampleMaterialTexture(specular_texture, uv, center_uv, angular_weight);
  vec3 normal = normalize(in_normal);
  vec3 view_direction = normalize(scene_uniforms.camera_position.xyz - in_world_position);
  vec3 result = albedo * scene_uniforms.ambient_color.rgb;

  if (scene_uniforms.directional_color.w > 0.0 &&
      dot(scene_uniforms.directional_direction.xyz, scene_uniforms.directional_direction.xyz) > 0.0) {
    vec3 directional_light = normalize(-scene_uniforms.directional_direction.xyz);
    result += blinnPhong(albedo, specular_color, normal, directional_light, view_direction,
                         scene_uniforms.directional_color.rgb,
                         scene_uniforms.directional_color.w);
  }

  int point_count = int(scene_uniforms.light_counts.x);
  for (int i = 0; i < point_count; ++i) {
    vec3 delta = point_lights.lights[i].position_intensity.xyz - in_world_position;
    float distance_squared = max(dot(delta, delta), 0.01);
    vec3 light_direction = normalize(delta);
    float attenuation = point_lights.lights[i].position_intensity.w / distance_squared;
    result += blinnPhong(albedo, specular_color, normal, light_direction, view_direction,
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
    result += blinnPhong(albedo, specular_color, normal, light_direction, view_direction,
                         light.color_outer_cos.rgb, attenuation);
  }

  result += sampleMaterialTexture(emissive_texture, uv, center_uv, angular_weight);
  out_color = vec4(clamp(result, 0.0, 1.0), 1.0);
}
