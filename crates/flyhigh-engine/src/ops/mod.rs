//! The kernels as plan steps: every `Gpu` method here records dispatches and returns the output tensor.

pub mod conv;
pub mod elementwise;
pub mod interp;
pub mod network;
pub mod params;
pub mod render;
pub mod step;
pub mod tensor;
pub mod texture;
