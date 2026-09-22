//! Android project intelligence — workspace, manifest, runtime, and capabilities.
//!
//! These modules provide the Android-aware layer of Grounding Coder 0.2:
//! runtime configuration (SDK versions, package, permissions), a registry of
//! device capabilities, and project workspace / manifest management.

pub mod capabilities;
pub mod project;
pub mod runtime;
