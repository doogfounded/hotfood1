use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// Each vertex: position (xyz), normal (xyz), uv (xy)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub pos:    [f32; 3],
    pub normal: [f32; 3],
    pub uv:     [f32; 2],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset:          0,
                    shader_location: 0,
                    format:          wgpu::VertexFormat::Float32x3,
                },
                // normal
                wgpu::VertexAttribute {
                    offset:          12,
                    shader_location: 1,
                    format:          wgpu::VertexFormat::Float32x3,
                },
                // uv
                wgpu::VertexAttribute {
                    offset:          24,
                    shader_location: 2,
                    format:          wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

pub struct RoomGeometry {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf:  wgpu::Buffer,
    pub index_count: u32,
}

// Room is a 10×5×10 box (half-extents 5, 2.5, 5)
const S: f32 = 5.0; // half-width/depth
const H: f32 = 5.0; // full height

fn quad(verts: &mut Vec<Vertex>, indices: &mut Vec<u16>,
        corners: [[f32; 3]; 4], normal: [f32; 3], uvs: [[f32; 2]; 4]) {
    let base = verts.len() as u16;
    for i in 0..4 {
        verts.push(Vertex { pos: corners[i], normal, uv: uvs[i] });
    }
    // Two triangles: 0-1-2 and 0-2-3
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}

pub fn create(device: &wgpu::Device) -> RoomGeometry {
    let mut verts:   Vec<Vertex> = Vec::new();
    let mut indices: Vec<u16>   = Vec::new();

    let uv = [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]];

    // Floor  (y=0, normal up)
    quad(&mut verts, &mut indices,
        [[-S,0.0,-S], [S,0.0,-S], [S,0.0,S], [-S,0.0,S]],
        [0.0, 1.0, 0.0], uv);

    // Ceiling (y=H, normal down)
    quad(&mut verts, &mut indices,
        [[-S,H,S], [S,H,S], [S,H,-S], [-S,H,-S]],
        [0.0,-1.0, 0.0], uv);

    // Back wall  (z=-S, normal +z)
    quad(&mut verts, &mut indices,
        [[-S,0.0,-S], [-S,H,-S], [S,H,-S], [S,0.0,-S]],
        [0.0, 0.0, 1.0], uv);

    // Front wall (z=+S, normal -z)
    quad(&mut verts, &mut indices,
        [[S,0.0,S], [S,H,S], [-S,H,S], [-S,0.0,S]],
        [0.0, 0.0,-1.0], uv);

    // Left wall  (x=-S, normal +x)
    quad(&mut verts, &mut indices,
        [[-S,0.0,S], [-S,H,S], [-S,H,-S], [-S,0.0,-S]],
        [1.0, 0.0, 0.0], uv);

    // Right wall (x=+S, normal -x)
    quad(&mut verts, &mut indices,
        [[S,0.0,-S], [S,H,-S], [S,H,S], [S,0.0,S]],
        [-1.0, 0.0, 0.0], uv);

    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("room_vertex"),
        contents: bytemuck::cast_slice(&verts),
        usage:    wgpu::BufferUsages::VERTEX,
    });
    let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("room_index"),
        contents: bytemuck::cast_slice(&indices),
        usage:    wgpu::BufferUsages::INDEX,
    });

    RoomGeometry { vertex_buf, index_buf, index_count: indices.len() as u32 }
}

pub fn render<'a>(geo: &'a RoomGeometry, pass: &mut wgpu::RenderPass<'a>) {
    pass.set_vertex_buffer(0, geo.vertex_buf.slice(..));
    pass.set_index_buffer(geo.index_buf.slice(..), wgpu::IndexFormat::Uint16);
    pass.draw_indexed(0..geo.index_count, 0, 0..1);
}
