// Room vertex + fragment shader.
// Uniform block matches CameraUniform in app.rs.
// The floor face (normal.y > 0.5) will sample the heat texture in Phase 3;
// for now it uses the same flat color path.

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexIn {
    @location(0) pos:    vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv:     vec2<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal:         vec3<f32>,
    @location(1) uv:             vec2<f32>,
    @location(2) world_pos:      vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos  = camera.view_proj * vec4<f32>(in.pos, 1.0);
    out.normal    = in.normal;
    out.uv        = in.uv;
    out.world_pos = in.pos;
    return out;
}

// Surface palette
fn surface_color(normal: vec3<f32>) -> vec3<f32> {
    // floor
    if normal.y >  0.5 { return vec3<f32>(0.22, 0.20, 0.18); }
    // ceiling
    if normal.y < -0.5 { return vec3<f32>(0.28, 0.28, 0.30); }
    // walls — slight warm/cool tint by axis
    if abs(normal.x) > 0.5 { return vec3<f32>(0.24, 0.22, 0.20); }
    return vec3<f32>(0.26, 0.24, 0.22);
}

// Simple directional + ambient light
fn light(normal: vec3<f32>, base: vec3<f32>) -> vec3<f32> {
    let light_dir = normalize(vec3<f32>(0.4, 1.0, 0.6));
    let diff      = max(dot(normal, light_dir), 0.0) * 0.4;
    let ambient   = 0.6;
    return base * (ambient + diff);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let base  = surface_color(in.normal);
    let color = light(in.normal, base);
    return vec4<f32>(color, 1.0);
}
