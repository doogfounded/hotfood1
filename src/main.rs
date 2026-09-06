mod app;
mod camera;
mod heat;
mod room;

use winit::event_loop::EventLoop;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut state = app::State::new_pending();
    event_loop.run_app(&mut state).unwrap();
}