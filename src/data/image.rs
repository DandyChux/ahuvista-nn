//! Image data processing module.
//!
//! Provides utilities for loading, resizing, and preprocessing
//! image data such as medical scans, photographs, or visualizations.

use crate::data::DataLoader;
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageReader, Rgb};
use ndarray::{Array3, Axis};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Structure for handling image data
pub struct ImageData {
    /// Processed images as 3D arrays (channels × height × width)
    pub images: Vec<Array3<f32>>,

    /// Image size after preprocessing
    pub image_size: (usize, usize),

    /// Number of channels (3 for RGB, 1 for grayscale)
    pub channels: usize,

    /// Mean pixel values for each channel
    pub channel_means: Vec<f32>,

    /// Standard deviation of pixel values for each channel
    pub channel_stds: Vec<f32>,

    /// Current index for batch iteration
    current_idx: usize,
}

impl ImageData {
    /// Create a new empty ImageData instance
    pub fn new() -> Self {
        ImageData {
            images: Vec::new(),
            image_size: (0, 0),
            channels: 0,
            channel_means: Vec::new(),
            channel_stds: Vec::new(),
            current_idx: 0,
        }
    }

    /// Load images from files
    ///
    /// # Arguments
    /// * `paths` - Paths to image files
    /// * `target_size` - Size to resize images to (height, width)
    /// * `grayscale` - Whether to convert images to grayscale
    ///
    /// # Returns
    /// Result containing ImageData or an error
    pub fn from_files<P: AsRef<Path>>(
        paths: &[P],
        target_size: (usize, usize),
        grayscale: bool,
    ) -> io::Result<Self> {
        let mut images = Vec::with_capacity(paths.len());
        let channels = if grayscale { 1 } else { 3 };

        // First pass: load and process images
        for path in paths {
            let img = ImageReader::open(path)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
                .decode()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

            let processed = if grayscale {
                let gray = img.to_luma8();
                let resized = image::imageops::resize(
                    &gray,
                    target_size.1 as u32,
                    target_size.0 as u32,
                    image::imageops::FilterType::Triangle,
                );

                // Convert to 3D array (1 × height × width)
                let mut array = Array3::zeros((1, target_size.0, target_size.1));

                for y in 0..target_size.0 {
                    for x in 0..target_size.1 {
                        array[[0, y, x]] = resized.get_pixel(x as u32, y as u32)[0] as f32;
                    }
                }

                array
            } else {
                let rgb = img.to_rgb8();
                let resized = image::imageops::resize(
                    &rgb,
                    target_size.1 as u32,
                    target_size.0 as u32,
                    image::imageops::FilterType::Triangle,
                );

                // Convert to 3D array (3 × height × width)
                let mut array = Array3::zeros((3, target_size.0, target_size.1));

                for y in 0..target_size.0 {
                    for x in 0..target_size.1 {
                        let pixel = resized.get_pixel(x as u32, y as u32);
                        array[[0, y, x]] = pixel[0] as f32;
                        array[[1, y, x]] = pixel[1] as f32;
                        array[[2, y, x]] = pixel[2] as f32;
                    }
                }

                array
            };

            images.push(processed);
        }

        // Calculate statistics for normalization
        let mut channel_sums = vec![0.0; channels];
        let mut channel_squares = vec![0.0; channels];
        let mut total_pixels = 0;

        for img in &images {
            total_pixels += img.shape()[1] * img.shape()[2]; // height × width

            for c in 0..channels {
                let channel_slice = img.index_axis(Axis(0), c);

                for &pixel in channel_slice.iter() {
                    channel_sums[c] += pixel;
                    channel_squares[c] += pixel * pixel;
                }
            }
        }

        // Calculate means and standard deviations
        let channel_means = channel_sums
            .iter()
            .map(|&sum| sum / (total_pixels as f32))
            .collect::<Vec<f32>>();

        let channel_stds = channel_squares
            .iter()
            .zip(&channel_means)
            .map(|(&sq_sum, &mean)| {
                let variance = sq_sum / (total_pixels as f32) - mean * mean;
                variance.sqrt().max(1e-6) // Avoid division by zero
            })
            .collect::<Vec<f32>>();

        // Normalize images
        let mut normalized_images = Vec::with_capacity(images.len());

        for img in &images {
            let mut normalized = img.clone();

            for c in 0..channels {
                let mut channel_slice = normalized.index_axis_mut(Axis(0), c);

                for pixel in channel_slice.iter_mut() {
                    *pixel = (*pixel - channel_means[c]) / channel_stds[c];
                }
            }

            normalized_images.push(normalized);
        }

        Ok(ImageData {
            images: normalized_images,
            image_size: target_size,
            channels,
            channel_means,
            channel_stds,
            current_idx: 0,
        })
    }

    /// Preprocess a new image (for inference)
    pub fn preprocess_image<P: AsRef<Path>>(&self, path: P) -> io::Result<Array3<f32>> {
        let img = ImageReader::open(path)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
            .decode()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let grayscale = self.channels == 1;

        let processed = if grayscale {
            let gray = img.to_luma8();
            let resized = image::imageops::resize(
                &gray,
                self.image_size.1 as u32,
                self.image_size.0 as u32,
                image::imageops::FilterType::Triangle,
            );

            // Convert to 3D array (1 × height × width)
            let mut array = Array3::zeros((1, self.image_size.0, self.image_size.1));

            for y in 0..self.image_size.0 {
                for x in 0..self.image_size.1 {
                    array[[0, y, x]] = resized.get_pixel(x as u32, y as u32)[0] as f32;
                }
            }

            array
        } else {
            let rgb = img.to_rgb8();
            let resized = image::imageops::resize(
                &rgb,
                self.image_size.1 as u32,
                self.image_size.0 as u32,
                image::imageops::FilterType::Triangle,
            );

            // Convert to 3D array (3 × height × width)
            let mut array = Array3::zeros((3, self.image_size.0, self.image_size.1));

            for y in 0..self.image_size.0 {
                for x in 0..self.image_size.1 {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    array[[0, y, x]] = pixel[0] as f32;
                    array[[1, y, x]] = pixel[1] as f32;
                    array[[2, y, x]] = pixel[2] as f32;
                }
            }

            array
        };

        // Normalize
        let mut normalized = processed.clone();

        for c in 0..self.channels {
            let mut channel_slice = normalized.index_axis_mut(Axis(0), c);

            for pixel in channel_slice.iter_mut() {
                *pixel = (*pixel - self.channel_means[c]) / self.channel_stds[c];
            }
        }

        Ok(normalized)
    }
}

impl DataLoader<Vec<Array3<f32>>> for ImageData {
    fn next_batch(&mut self, batch_size: usize) -> Option<Vec<Array3<f32>>> {
        if self.current_idx >= self.images.len() {
            return None;
        }

        let end_idx = (self.current_idx + batch_size).min(self.images.len());
        let batch = self.images[self.current_idx..end_idx].to_vec();

        self.current_idx = end_idx;
        Some(batch)
    }

    fn reset(&mut self) {
        self.current_idx = 0;
    }

    fn len(&self) -> usize {
        self.images.len()
    }
}
