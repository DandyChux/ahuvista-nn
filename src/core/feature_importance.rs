use super::Module;
use crate::data::MultiModalBatch;
use crate::training::metrics::{EvaluationMetrics, evalute_auroc};
use rand::rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// Calculates feature importance by measuring performance drop when a feature is shuffled.
pub fn calculate_permutation_importance<I, O>(
    model: &mut impl Module<I, O>,         // A generic model
    validation_data: &mut MultiModalBatch, // A batch of validation data
    baseline_metrics: &EvaluationMetrics,
) -> HashMap<String, f32> {
    let mut importances = HashMap::new();
    let feature_names = ["age", "blood_pressure", "bmi"]; // You'd get this from your StaticFeatureData struct

    for (i, feature_name) in feature_names.iter().enumerate() {
        let mut feature_column: Vec<f32> = validation_data
            .sequences
            .iter()
            .map(|sequence| sequence.static_features.get(i).copied().unwrap_or(0.0))
            .collect();

        feature_column.shuffle(&mut rng());

        for (sequence, value) in validation_data
            .sequences
            .iter_mut()
            .zip(feature_column.into_iter())
        {
            if i < sequence.static_features.len() {
                sequence.static_features[i] = value;
            }
        }

        let new_auroc = evalute_auroc::<I, O>(model, validation_data);

        let importance = baseline_metrics.auroc - new_auroc;
        importances.insert(feature_name.to_string(), importance);
    }

    importances
}
