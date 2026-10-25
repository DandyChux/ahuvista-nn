//! Data preprocessing and splitting utilities

use crate::{
    data::{FeatureStats, PreparedDataset},
    error::{AppError, AppResult},
    models::PatientSequence,
};
use ndarray::Array3;
use tracing::info;

pub fn split_dataset(
    dataset: &PreparedDataset,
    train_ratio: f32,
) -> AppResult<(PreparedDataset, PreparedDataset)> {
    if !(0.0..=1.0).contains(&train_ratio) {
        return Err(AppError::InvalidData(format!(
            "Train ratio must be between 0 and 1, got {}",
            train_ratio
        )));
    }

    let split_index = (dataset.len() as f32 * train_ratio).floor() as usize;

    dataset.split_at(split_index)
}

fn split_vec<T: Clone>(values: &[T], split_index: usize) -> (Vec<T>, Vec<T>) {
    (
        values[..split_index].to_vec(),
        values[split_index..].to_vec(),
    )
}

fn split_slice<T: Clone>(values: &[T], split_index: usize) -> (Vec<T>, Vec<T>) {
    split_vec(values, split_index)
}

/// Extract demographic strata from unified patient sequences.
///
/// This currently uses deterministic placeholder strata. Replace the
/// feature-index logic once the static feature schema is finalized.
pub fn extract_patient_strata(sequences: &[PatientSequence]) -> Vec<String> {
    sequences
        .iter()
        .enumerate()
        .map(|(index, _)| match index % 4 {
            0 => "urban_young".to_string(),
            1 => "urban_old".to_string(),
            2 => "rural_young".to_string(),
            _ => "rural_old".to_string(),
        })
        .collect()
}

/// Normalize static and dynamic features in-place.
pub fn normalize_sequences(
    sequences: &mut [PatientSequence],
    static_stats: &[FeatureStats],
    dynamic_stats: &[FeatureStats],
) {
    for sequence in sequences {
        normalize_static_features(&mut sequence.static_features, static_stats);

        normalize_dynamic_features(&mut sequence.dynamic_features, dynamic_stats);
    }
}

fn normalize_static_features(features: &mut [f32], stats: &[FeatureStats]) {
    for (index, feature) in features.iter_mut().enumerate() {
        let Some(stat) = stats.get(index) else {
            continue;
        };

        *feature = normalize_value(*feature, stat);
    }
}

fn normalize_dynamic_features(sequence: &mut [Vec<f32>], stats: &[FeatureStats]) {
    for timestep in sequence {
        for (index, feature) in timestep.iter_mut().enumerate() {
            let Some(stat) = stats.get(index) else {
                continue;
            };

            *feature = normalize_value(*feature, stat);
        }
    }
}

fn normalize_value(value: f32, stat: &FeatureStats) -> f32 {
    match stat.std.is_finite() && stat.std > 1e-6 {
        true => (value - stat.mean) / stat.std,
        false => value - stat.mean,
    }
}
