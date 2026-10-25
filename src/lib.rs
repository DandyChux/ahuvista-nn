//! Ahuvista-NN: Multi-modal Neural Network for Maternal Mortality Prediction
//!
//! This library provides components for building, training, and evaluating
//! multi-modal neural networks specialized for healthcare risk prediction.

// Core neural network components
pub mod core;

// Data handling and loaders for different modalities
pub mod data;

// Model architectures for specific modalities
pub mod models;

// Training, loss functions, and optimization
pub mod training;

// Config
pub mod config;

// Audit
pub mod audit;

pub mod error;

pub mod application;

pub mod prediction;

pub mod model_artifact;
