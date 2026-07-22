//! Data handling utilities for different modalities.
//!
//! This module provides tools for loading, preprocessing, and transforming
//! data from different sources (tabular, temporal, text, image) to be used
//! with the multi-modal neural network.

pub mod db;
#[cfg(feature = "fhir")]
pub mod fhir;
pub mod image;
pub mod loader;
pub mod preprocessing;
pub mod tabular;
pub mod temporal;
pub mod text;

#[cfg(feature = "fhir")]
pub use fhir::*;
pub use image::ImageData;
pub use loader::*;
pub use tabular::TabularData;
pub use temporal::TemporalData;
pub use text::TextData;
