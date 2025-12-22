//! Data preprocessing and splitting utilities

use super::TabularData;
use ndarray::Array3;
use tracing::info;

/// Split multi-modal data with sample weights into training and validation sets
pub fn split_data_with_weights(
    tabular_data: &[Vec<f32>],
    temporal_data: &[Vec<Vec<f32>>],
    text_data: &[Vec<usize>],
    image_data: &[Array3<f32>],
    targets: &[f32],
    weights: &[f32],
    train_ratio: f32,
) -> (
    (
        Vec<Vec<f32>>,
        Vec<Vec<Vec<f32>>>,
        Vec<Vec<usize>>,
        Vec<Array3<f32>>,
    ),
    (
        Vec<Vec<f32>>,
        Vec<Vec<Vec<f32>>>,
        Vec<Vec<usize>>,
        Vec<Array3<f32>>,
    ),
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    let total_samples = targets.len();
    let train_size = (total_samples as f32 * train_ratio) as usize;
    info!("TRAINING SIZE: {}", train_size);
    info!("TABULAR DATA LENGTH: {}", tabular_data.len());
    info!("TEMPORAL DATA LENGTH: {}", temporal_data.len());
    info!("TEXT DATA LENGTH: {}", text_data.len());
    info!("IMAGE DATA LENGTH: {}", image_data.len());

    // Split tabular data (only if it has data)
    let (train_tabular, val_tabular) = if !tabular_data.is_empty() {
        (
            tabular_data[0..train_size].to_vec(),
            tabular_data[train_size..].to_vec(),
        )
    } else {
        (vec![], vec![])
    };

    // Split temporal data (only if it has data)
    let (train_temporal, val_temporal) = if !temporal_data.is_empty() {
        (
            temporal_data[0..train_size].to_vec(),
            temporal_data[train_size..].to_vec(),
        )
    } else {
        (vec![], vec![])
    };

    // Split text data (only if it has data)
    let (train_text, val_text) = if !text_data.is_empty() {
        (
            text_data[0..train_size].to_vec(),
            text_data[train_size..].to_vec(),
        )
    } else {
        (vec![], vec![])
    };

    // Split image data (only if it has data)
    let (train_image, val_image) = if !image_data.is_empty() {
        (
            image_data[0..train_size].to_vec(),
            image_data[train_size..].to_vec(),
        )
    } else {
        (vec![], vec![])
    };

    // Split targets
    let train_targets = targets[0..train_size].to_vec();
    let val_targets = targets[train_size..].to_vec();

    // Split weights
    let train_weights = weights[0..train_size].to_vec();
    let val_weights = weights[train_size..].to_vec();

    (
        (train_tabular, train_temporal, train_text, train_image),
        (val_tabular, val_temporal, val_text, val_image),
        train_targets,
        val_targets,
        train_weights,
        val_weights,
    )
}

/// Extract patient demographic strata from tabular data
pub fn extract_patient_strata(tabular_data: &TabularData) -> Vec<String> {
    // This is a placeholder function - in a real implementation,
    // you would extract demographic features from the tabular data
    // to determine each patient's stratum

    let mut strata = Vec::with_capacity(tabular_data.data.len());

    for (i, _) in tabular_data.data.iter().enumerate() {
        // This is just a dummy implementation
        let stratum = if i % 4 == 0 {
            "urban_young".to_string()
        } else if i % 4 == 1 {
            "urban_old".to_string()
        } else if i % 4 == 2 {
            "rural_young".to_string()
        } else {
            "rural_old".to_string()
        };

        strata.push(stratum);
    }

    strata
}
