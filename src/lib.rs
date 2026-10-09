//! lookfocus: switch Hyprland monitor focus by where you are facing.
//!
//! The pipeline is camera frame, face landmarks, head pose, smoothing, then
//! monitor classification. Each stage lives in its own module and has no
//! knowledge of the stages around it, so each can be tested on its own.

pub mod adaptive;
pub mod calibrate;
pub mod camera;
pub mod classify;
pub mod config;
pub mod control;
pub mod daemon;
pub mod events;
pub mod filter;
pub mod gesture;
pub mod hypr;
pub mod image;
pub mod models;
pub mod notify;
pub mod pose;
pub mod sampler;
pub mod state;
pub mod switcher;
pub mod vision;
