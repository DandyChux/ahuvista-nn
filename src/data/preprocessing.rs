//! Data preprocessing and splitting utilities

use super::TabularData;
use ndarray::Array3;

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

    // Split tabular data
    let train_tabular = tabular_data[0..train_size].to_vec();
    let val_tabular = tabular_data[train_size..].to_vec();

    // Split temporal data
    let train_temporal = temporal_data[0..train_size].to_vec();
    let val_temporal = temporal_data[train_size..].to_vec();

    // Split text data
    let train_text = text_data[0..train_size].to_vec();
    let val_text = text_data[train_size..].to_vec();

    // Split image data
    let train_image = image_data[0..train_size].to_vec();
    let val_image = image_data[train_size..].to_vec();

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
