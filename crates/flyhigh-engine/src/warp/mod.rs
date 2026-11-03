//! Camera warp (ARCHITECTURE.md): a frame is turned by the camera motion the host's input says happened after the
//! frame was made, so a mouse-driven camera answers at once although the picture comes late. Only where the mouse turns
//! the camera: the calibrator finds that out from the input and the measured image motion, and holds the warp off
//! otherwise.
pub mod calib;
pub mod gpu;

pub use calib::{CalibOptions, Calibration, InputCalibrator};
pub use gpu::{CameraWarp, WarpOptions};
