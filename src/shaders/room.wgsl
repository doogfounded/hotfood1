// Room vertex + fragment shader.
// group(0): camera uniform
// group(1): heat texture + sampler (floor fragments only)

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var heat_tex:     texture_2d<f32>;
@group(1) @binding(1)
var heat_sampler: sampler;

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

// ── Surface palette ───────────────────────────────────────────────────────────

fn surface_color(normal: vec3<f32>) -> vec3<f32> {
    if normal.y >  0.5 { return vec3<f32>(0.18, 0.16, 0.14); }   // floor — dark stone
    if normal.y < -0.5 { return vec3<f32>(0.26, 0.26, 0.28); }   // ceiling
    if abs(normal.x) > 0.5 { return vec3<f32>(0.22, 0.20, 0.19); } // side walls
    return vec3<f32>(0.24, 0.22, 0.20);                            // front/back walls
}

fn light(normal: vec3<f32>, base: vec3<f32>) -> vec3<f32> {
    let light_dir = normalize(vec3<f32>(0.4, 1.0, 0.6));
    let diff      = max(dot(normal, light_dir), 0.0) * 0.35;
    return base * (0.65 + diff);
}

// ── Thermal color ramp ────────────────────────────────────────────────────────
// deep red → orange → yellow → near-white

fn heat_color(t: f32) -> vec3<f32> {
    let a = vec3<f32>(0.55, 0.02, 0.02);   // deep red
    let b = vec3<f32>(1.00, 0.28, 0.00);   // orange
    let c = vec3<f32>(1.00, 0.85, 0.05);   // yellow
    let d = vec3<f32>(1.00, 0.98, 0.88);   // white-hot

    if t < 0.33 {
        return mix(a, b, t / 0.33);
    } else if t < 0.66 {
        return mix(b, c, (t - 0.33) / 0.33);
    } else {
        return mix(c, d, (t - 0.66) / 0.34);
    }
}

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let base  = surface_color(in.normal);
    var color = light(in.normal, base);

    // Heat overlay — floor only (normal points up)
    if in.normal.y > 0.5 {
        let heat = textureSample(heat_tex, heat_sampler, in.uv).r;
        if heat > 0.005 {
            let hc    = heat_color(heat);
            // Blend: low heat = subtle glow, high heat = dominant
            let blend = smoothstep(0.0, 0.25, heat);
            color     = mix(color, hc, blend * 0.92);
            // Emissive boost so hot spots actually glow
            color    += hc * heat * 0.18;
        }
    }

    return vec4<f32>(color, 1.0);
}
