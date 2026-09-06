use glam::{Mat4, Vec3};
use std::f32::consts::FRAC_PI_2;

pub struct Camera {
    pub position: Vec3,
    pub yaw:      f32,  // radians, horizontal
    pub pitch:    f32,  // radians, vertical (clamped)
    pub fov_y:    f32,  // radians
    pub aspect:   f32,
}

// Half-extents of the room the camera is clamped inside
const ROOM_HALF: f32 = 4.5;
const ROOM_Y_MIN: f32 = 0.3;
const ROOM_Y_MAX: f32 = 4.7;

pub fn create(aspect: f32) -> Camera {
    Camera {
        position: Vec3::new(0.0, 1.6, 0.0),
        yaw:      0.0,
        pitch:    0.0,
        fov_y:    60_f32.to_radians(),
        aspect,
    }
}

pub fn update(cam: &mut Camera, move_input: Vec3, mouse_delta: (f32, f32), dt: f32) {
    // Mouse look
    let sensitivity = 0.002_f32;
    cam.yaw   -= mouse_delta.0 * sensitivity;
    cam.pitch -= mouse_delta.1 * sensitivity;
    cam.pitch  = cam.pitch.clamp(-FRAC_PI_2 + 0.05, FRAC_PI_2 - 0.05);

    // Build forward/right from yaw only (no pitch in movement)
    let (sy, cy) = cam.yaw.sin_cos();
    let forward  = Vec3::new(-sy, 0.0, -cy).normalize_or_zero();
    let right    = Vec3::new(cy, 0.0, -sy).normalize_or_zero();

    let speed = 3.0_f32;
    cam.position += forward * move_input.z * speed * dt;
    cam.position += right   * move_input.x * speed * dt;
    cam.position.y += move_input.y * speed * dt;

    // Clamp inside room
    cam.position.x = cam.position.x.clamp(-ROOM_HALF, ROOM_HALF);
    cam.position.y = cam.position.y.clamp(ROOM_Y_MIN, ROOM_Y_MAX);
    cam.position.z = cam.position.z.clamp(-ROOM_HALF, ROOM_HALF);
}

pub fn view_proj(cam: &Camera) -> Mat4 {
    let (sp, cp) = cam.pitch.sin_cos();
    let (sy, cy) = cam.yaw.sin_cos();
    let look_dir = Vec3::new(cp * (-sy), sp, cp * (-cy));
    let view     = Mat4::look_to_rh(cam.position, look_dir, Vec3::Y);
    let proj     = Mat4::perspective_rh(cam.fov_y, cam.aspect, 0.05, 50.0);
    proj * view
}

pub fn set_aspect(cam: &mut Camera, aspect: f32) {
    cam.aspect = aspect;
}
