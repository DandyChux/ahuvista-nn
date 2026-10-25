//! Data handling utilities for different modalities.
//!
//! This module provides tools for loading, preprocessing, and transforming
//! data from different sources (tabular, temporal, text, image) to be used
//! with the multi-modal neural network.

pub mod db;
pub mod fhir;
pub mod image;
pub mod ingestion;
pub mod loader;
pub mod migrations;
pub mod normalization;
pub mod preprocessing;
pub mod sequence;
pub mod source;
pub mod text;

pub use fhir::*;
pub use image::ImageData;
pub use ingestion::*;
pub use loader::*;
pub use migrations::*;
pub use normalization::*;
pub use sequence::*;
pub use source::*;
pub use text::TextData;
