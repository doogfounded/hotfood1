// CPU-side heat simulation — 64×64 R8Unorm texture updated every frame.
// Diffusion and decay happen on the CPU (4096 cells = trivially fast),
// then the result is uploaded to GPU via queue.write_texture.

const SIZE: u32  = 64;
const S: usize   = SIZE as usize;
const CELLS: usize = S * S;

// Tuning knobs
const DIFFUSION: f32 = 2.0;   // lateral spread speed (units/sec)
const DECAY:     f32 = 0.35;  // fraction of heat lost per second

pub struct HeatMap {
    data:    Vec<f32>,   // current heat [0,1]
    temp:    Vec<f32>,   // scratch buffer for diffusion step
    pub texture: wgpu::Texture,
    pub view:    wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

pub fn create(device: &wgpu::Device) -> HeatMap {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("heat_tex"),
        size:            wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::R8Unorm,
        usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats:    &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label:            Some("heat_sampler"),
        address_mode_u:   wgpu::AddressMode::ClampToEdge,
        address_mode_v:   wgpu::AddressMode::ClampToEdge,
        address_mode_w:   wgpu::AddressMode::ClampToEdge,
        mag_filter:       wgpu::FilterMode::Linear,
        min_filter:       wgpu::FilterMode::Linear,
        mipmap_filter:    wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    HeatMap { data: vec![0.0; CELLS], temp: vec![0.0; CELLS], texture, view, sampler }
}

/// Write a radial heat blob onto the floor.
/// `norm_x`, `norm_z` are UV coordinates in [0, 1].
/// `radius` is in UV space (e.g. 0.08 = roughly one object footprint).
pub fn stamp(heat: &mut HeatMap, norm_x: f32, norm_z: f32, radius: f32, intensity: f32) {
    let cx = (norm_x * S as f32) as i32;
    let cz = (norm_z * S as f32) as i32;
    let r  = ((radius * S as f32) as i32).max(1);

    for dz in -r..=r {
        for dx in -r..=r {
            let px = cx + dx;
            let pz = cz + dz;
            if px < 0 || pz < 0 || px >= S as i32 || pz >= S as i32 { continue; }
            let dist_sq = (dx * dx + dz * dz) as f32 / (r * r) as f32;
            if dist_sq > 1.0 { continue; }
            let falloff = 1.0 - dist_sq;     // quadratic falloff
            let idx = pz as usize * S + px as usize;
            heat.data[idx] = (heat.data[idx] + intensity * falloff).min(1.0);
        }
    }
}

/// 5-tap diffusion + exponential decay, dt-scaled.
pub fn diffuse(heat: &mut HeatMap, dt: f32) {
    let diff  = (DIFFUSION * dt * 0.25).min(0.24); // weight on neighbors
    let decay = (1.0 - DECAY * dt).max(0.0);

    for z in 0..S {
        for x in 0..S {
            let c = heat.data[z * S + x];
            let l = if x > 0     { heat.data[z * S + x - 1] } else { c };
            let r = if x < S - 1 { heat.data[z * S + x + 1] } else { c };
            let u = if z > 0     { heat.data[(z - 1) * S + x] } else { c };
            let d = if z < S - 1 { heat.data[(z + 1) * S + x] } else { c };
            let spread = (l + r + u + d - 4.0 * c) * diff;
            heat.temp[z * S + x] = ((c + spread) * decay).clamp(0.0, 1.0);
        }
    }
    std::mem::swap(&mut heat.data, &mut heat.temp);
}

/// Upload current heat data to the GPU texture.
pub fn upload(heat: &HeatMap, queue: &wgpu::Queue) {
    let bytes: Vec<u8> = heat.data.iter().map(|v| (v * 255.0) as u8).collect();
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture:   &heat.texture,
            mip_level: 0,
            origin:    wgpu::Origin3d::ZERO,
            aspect:    wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::ImageDataLayout {
            offset:         0,
            bytes_per_row:  Some(SIZE),
            rows_per_image: Some(SIZE),
        },
        wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
    );
}
