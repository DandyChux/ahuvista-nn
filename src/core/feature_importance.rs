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
    let feature_names = ["age", "blood_pressure", "bmi"]; // You'd get this from your TabularData struct

    for (i, feature_name) in feature_names.iter().enumerate() {
        let perturbed_data: &mut MultiModalBatch = validation_data;

        // Shuffle the i-th column (feature) of the tabular data
        let mut feature_column: Vec<_> = perturbed_data.tabular.iter().map(|row| row[i]).collect();
        feature_column.shuffle(&mut rng());

        for (j, row) in perturbed_data.tabular.iter_mut().enumerate() {
            row[i] = feature_column[j];
        }

        // Re-evaluate the model on the perturbed data
        // This requires a simplified evaluation function that just returns one metric, e.g., AUROC
        // Note: evaluate_auroc needs to be implemented or imported
        let new_auroc = evalute_auroc::<I, O>(model, perturbed_data);

        // Importance is the drop in performance
        let importance = baseline_metrics.auroc - new_auroc;
        importances.insert(feature_name.to_string(), importance);
    }

    importances
}
