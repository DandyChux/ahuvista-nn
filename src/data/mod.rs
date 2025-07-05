//! Data handling utilities for different modalities.
//!
//! This module provides tools for loading, preprocessing, and transforming
//! data from different sources (tabular, temporal, text, image) to be used
//! with the multi-modal neural network.

pub mod image;
pub mod tabular;
pub mod temporal;
pub mod text;

pub use image::ImageData;
pub use tabular::TabularData;
pub use temporal::TemporalData;
pub use text::TextData;

/// Common trait for data loaders that can provide batches of data
pub trait DataLoader<T> {
    /// Get the next batch of data
    fn next_batch(&mut self, batch_size: usize) -> Option<T>;

    /// Reset the data loader to start from the beginning
    fn reset(&mut self);

    /// Get the total number of samples
    fn len(&self) -> usize;

    /// Check if there are no samples
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Structure holding a batch of multi-modal data
pub struct MultiModalBatch {
    /// Tabular features
    pub tabular: Vec<Vec<f32>>,

    /// Temporal sequences
    pub temporal: Vec<Vec<Vec<f32>>>,

    /// Text token sequences
    pub text: Vec<Vec<usize>>,

    /// Image data
    pub images: Vec<ndarray::Array3<f32>>,

    /// Target values (0.0 or 1.0 for binary classification)
    pub targets: Vec<f32>,
}

/// Combined data loader for multi-modal data
pub struct MultiModalDataLoader {
    /// Tabular data
    pub tabular_data: TabularData,

    /// Temporal data
    pub temporal_data: TemporalData,

    /// Text data
    pub text_data: TextData,

    /// Image data
    pub image_data: ImageData,

    /// Target values
    pub targets: Vec<f32>,

    /// Current position in the dataset
    current_idx: usize,
}

impl MultiModalDataLoader {
    /// Create a new multi-modal data loader
    pub fn new(
        tabular_data: TabularData,
        temporal_data: TemporalData,
        text_data: TextData,
        image_data: ImageData,
        targets: Vec<f32>,
    ) -> Self {
        // Ensure all data sources have the same number of samples
        assert_eq!(
            tabular_data.data.len(),
            targets.len(),
            "Tabular data and targets must have same length"
        );
        assert_eq!(
            temporal_data.sequences.len(),
            targets.len(),
            "Temporal data and targets must have same length"
        );
        assert_eq!(
            text_data.documents.len(),
            targets.len(),
            "Text data and targets must have same length"
        );
        assert_eq!(
            image_data.images.len(),
            targets.len(),
            "Image data and targets must have same length"
        );

        MultiModalDataLoader {
            tabular_data,
            temporal_data,
            text_data,
            image_data,
            targets,
            current_idx: 0,
        }
    }
}

impl DataLoader<MultiModalBatch> for MultiModalDataLoader {
    fn next_batch(&mut self, batch_size: usize) -> Option<MultiModalBatch> {
        if self.current_idx >= self.targets.len() {
            return None;
        }

        let end_idx = (self.current_idx + batch_size).min(self.targets.len());
        let batch_size = end_idx - self.current_idx;

        let mut batch = MultiModalBatch {
            tabular: Vec::with_capacity(batch_size),
            temporal: Vec::with_capacity(batch_size),
            text: Vec::with_capacity(batch_size),
            images: Vec::with_capacity(batch_size),
            targets: Vec::with_capacity(batch_size),
        };

        for i in self.current_idx..end_idx {
            batch.tabular.push(self.tabular_data.data[i].clone());
            batch.temporal.push(self.temporal_data.sequences[i].clone());
            batch.text.push(self.text_data.token_sequences[i].clone());
            batch.images.push(self.image_data.images[i].clone());
            batch.targets.push(self.targets[i]);
        }

        self.current_idx = end_idx;
        Some(batch)
    }

    fn reset(&mut self) {
        self.current_idx = 0;
    }

    fn len(&self) -> usize {
        self.targets.len()
    }
}
