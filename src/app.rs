use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use crate::{camera, heat, room};

// ── Camera uniform (must match room.wgsl) ────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

fn camera_uniform(mat: Mat4) -> CameraUniform {
    CameraUniform { view_proj: mat.to_cols_array_2d() }
}

// ── Input state ───────────────────────────────────────────────────────────────

#[derive(Default)]
struct InputState {
    w: bool, a: bool, s: bool, d: bool,
    space: bool, shift: bool,
    mouse_dx: f32, mouse_dy: f32,
}

impl InputState {
    fn move_vec(&self) -> glam::Vec3 {
        let x = (self.d as i32 - self.a as i32) as f32;
        let y = (self.space as i32 - self.shift as i32) as f32;
        let z = (self.w as i32 - self.s as i32) as f32;
        glam::Vec3::new(x, y, z)
    }

    fn consume_mouse(&mut self) -> (f32, f32) {
        let delta = (self.mouse_dx, self.mouse_dy);
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        delta
    }
}

// ── Live wgpu state ───────────────────────────────────────────────────────────

struct GfxState {
    window:        Arc<Window>,
    surface:       wgpu::Surface<'static>,
    device:        wgpu::Device,
    queue:         wgpu::Queue,
    surface_cfg:   wgpu::SurfaceConfiguration,
    depth_texture: wgpu::Texture,
    depth_view:    wgpu::TextureView,

    // Room rendering
    room_geo:        room::RoomGeometry,
    render_pipeline: wgpu::RenderPipeline,
    camera_buf:      wgpu::Buffer,
    camera_bg:       wgpu::BindGroup,

    // Heat
    heat_map: heat::HeatMap,
    heat_bg:  wgpu::BindGroup,

    // Per-frame state
    camera:        camera::Camera,
    input:         InputState,
    last_frame:    std::time::Instant,
    cursor_locked: bool,
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32)
    -> (wgpu::Texture, wgpu::TextureView)
{
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size:  wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Depth32Float,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats:    &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

impl GfxState {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(Arc::clone(&window)).unwrap();

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference:       wgpu::PowerPreference::HighPerformance,
            compatible_surface:     Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label:    Some("device"),
            required_features: wgpu::Features::empty(),
            required_limits:   wgpu::Limits::default(),
            ..Default::default()
        }, None).await.unwrap();

        let surface_caps   = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_cfg = wgpu::SurfaceConfiguration {
            usage:        wgpu::TextureUsages::RENDER_ATTACHMENT,
            format:       surface_format,
            width:        size.width,
            height:       size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode:   surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_cfg);

        let (depth_texture, depth_view) =
            create_depth_texture(&device, size.width, size.height);

        // Camera uniform buffer
        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("camera_buf"),
            contents: bytemuck::bytes_of(&camera_uniform(Mat4::IDENTITY)),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind group layout: just the camera uniform
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty:         wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("camera_bg"),
            layout:  &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        // Heat bind group layout
        let heat_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("heat_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled:   false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Render pipeline
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/room.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("room_pl"),
            bind_group_layouts:   &[&bgl, &heat_bgl],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("room_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: Some("vs_main"),
                buffers:     &[room::Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     surface_format,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:          wgpu::PrimitiveTopology::TriangleList,
                front_face:        wgpu::FrontFace::Ccw,
                cull_mode:         Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare:       wgpu::CompareFunction::Less,
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview:   None,
            cache:        None,
        });

        let room_geo = room::create(&device);

        // Heat map — pre-stamp a couple of blobs so something is visible immediately
        let mut heat_map = heat::create(&device);
        heat::stamp(&mut heat_map, 0.5,  0.5,  0.12, 1.0);  // center
        heat::stamp(&mut heat_map, 0.25, 0.3,  0.08, 0.7);  // left
        heat::stamp(&mut heat_map, 0.72, 0.65, 0.06, 0.5);  // right
        heat::upload(&heat_map, &queue);

        let heat_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("heat_bg"),
            layout:  &heat_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(&heat_map.view),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::Sampler(&heat_map.sampler),
                },
            ],
        });

        let aspect = size.width as f32 / size.height.max(1) as f32;
        let cam    = camera::create(aspect);

        Self {
            window,
            surface,
            device,
            queue,
            surface_cfg,
            depth_texture,
            depth_view,
            room_geo,
            render_pipeline,
            camera_buf,
            camera_bg,
            heat_map,
            heat_bg,
            camera: cam,
            input:  InputState::default(),
            last_frame: std::time::Instant::now(),
            cursor_locked: false,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.surface_cfg.width  = width;
        self.surface_cfg.height = height;
        self.surface.configure(&self.device, &self.surface_cfg);
        let (dt, dv) = create_depth_texture(&self.device, width, height);
        self.depth_texture = dt;
        self.depth_view    = dv;
        camera::set_aspect(&mut self.camera, width as f32 / height as f32);
    }

    fn render(&mut self) {
        let now = std::time::Instant::now();
        let dt  = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        // Update camera
        let mouse = self.input.consume_mouse();
        let mv    = self.input.move_vec();
        if self.cursor_locked {
            camera::update(&mut self.camera, mv, mouse, dt);
        }

        // Diffuse + upload heat
        heat::diffuse(&mut self.heat_map, dt);
        heat::upload(&self.heat_map, &self.queue);

        // Upload camera uniform
        let vp  = camera::view_proj(&self.camera);
        let uni = camera_uniform(vp);
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&uni));

        // Get surface texture
        let output  = match self.surface.get_current_texture() {
            Ok(t)  => t,
            Err(_) => return,
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("frame") });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("room_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05, g: 0.05, b: 0.07, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view:        &self.depth_view,
                    depth_ops:   Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.camera_bg, &[]);
            pass.set_bind_group(1, &self.heat_bg, &[]);
            room::render(&self.room_geo, &mut pass);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn lock_cursor(&mut self, locked: bool) {
        self.cursor_locked = locked;
        if locked {
            let _ = self.window.set_cursor_grab(CursorGrabMode::Confined);
            self.window.set_cursor_visible(false);
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
            self.window.set_cursor_visible(true);
        }
    }
}

// ── ApplicationHandler (winit 0.30) ──────────────────────────────────────────

pub struct State {
    gfx: Option<GfxState>,
}

impl State {
    pub fn new_pending() -> Self {
        Self { gfx: None }
    }
}

impl ApplicationHandler for State {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() { return; }
        let attrs  = Window::default_attributes()
            .with_title("A Room That Remembers Heat")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.gfx   = Some(pollster::block_on(GfxState::new(window)));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId,
                    event: WindowEvent) {
        let gfx = match &mut self.gfx { Some(g) => g, None => return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                gfx.resize(size.width, size.height);
            }

            WindowEvent::RedrawRequested => {
                gfx.render();
                gfx.window.request_redraw();
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Left
                    && state == ElementState::Pressed
                {
                    gfx.lock_cursor(true);
                }
            }

            WindowEvent::Focused(false) => {
                gfx.lock_cursor(false);
            }

            WindowEvent::KeyboardInput { event: KeyEvent { physical_key, state, .. }, .. } => {
                let pressed = state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = physical_key {
                    match code {
                        KeyCode::KeyW     => gfx.input.w     = pressed,
                        KeyCode::KeyA     => gfx.input.a     = pressed,
                        KeyCode::KeyS     => gfx.input.s     = pressed,
                        KeyCode::KeyD     => gfx.input.d     = pressed,
                        KeyCode::Space    => gfx.input.space  = pressed,
                        KeyCode::ShiftLeft | KeyCode::ShiftRight
                                          => gfx.input.shift  = pressed,
                        KeyCode::Escape   => {
                            if pressed { gfx.lock_cursor(false); }
                        }
                        _ => {}
                    }
                }
            }

            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop,
                    _device_id: winit::event::DeviceId,
                    event: DeviceEvent) {
        let gfx = match &mut self.gfx { Some(g) => g, None => return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            gfx.input.mouse_dx += dx as f32;
            gfx.input.mouse_dy += dy as f32;
        }
    }
}
