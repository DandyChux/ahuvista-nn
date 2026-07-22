use crate::{core::Module, data::MultiModalBatch};

pub struct ConfusionMatrix {
    /// True Positives
    pub tp: u32,
    /// False Positives
    pub fp: u32,
    /// True Negatives
    pub tn: u32,
    /// False Negatives
    pub fn_: u32,
}

impl ConfusionMatrix {
    pub fn from_predictions(predictions: &[f32], targets: &[f32], threshold: f32) -> Self {
        let mut conf = Self {
            tp: 0,
            fp: 0,
            tn: 0,
            fn_: 0,
        };
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let p_class = if *pred >= threshold { 1.0 } else { 0.0 };
            if p_class == 1.0 && *target == 1.0 {
                conf.tp += 1;
            } else if p_class == 1.0 && *target == 0.0 {
                conf.fp += 1;
            } else if p_class == 0.0 && *target == 0.0 {
                conf.tn += 1;
            } else if p_class == 0.0 && *target == 1.0 {
                conf.fn_ += 1;
            }
        }
        conf
    }

    // Positive Predictive Value (Precision)
    pub fn ppv(&self) -> f32 {
        if self.tp + self.fp == 0 {
            0.0
        } else {
            self.tp as f32 / (self.tp + self.fp) as f32
        }
    }

    // Negative Predictive Value
    pub fn npv(&self) -> f32 {
        if self.tn + self.fn_ == 0 {
            0.0
        } else {
            self.tn as f32 / (self.tn + self.fn_) as f32
        }
    }

    // Other metrics like sensitivity, specificity...
}

#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    /// Loss
    pub loss: f32,
    /// Accuracy
    pub accuracy: f32,
    /// Positive Predictive Value (Precision)
    pub ppv: f32,
    /// Negative Predictive Value
    pub npv: f32,
    /// Recall or True Positive Rate
    pub sensitivity: f32,
    /// True negative rate
    pub specificity: f32,
    /// Area Under Receiver Operating Characteristic Curve
    pub auroc: f32,
    /// Area Under Precision-Recall Curve
    pub auprc: f32,
    /// Mean of predicted values
    pub mean_pred: f32,
    /// Standard deviation of predicted values
    pub std_pred: f32,
    /// Harmonic mean of precision and recall
    pub f1_score: f32,
    /// Matthews Correlation Coefficient
    pub mcc: f32,
    /// Average of sensitivity and specificity
    pub balanced_accuracy: f32,
}

impl Default for EvaluationMetrics {
    fn default() -> Self {
        Self {
            loss: 0.0,
            accuracy: 0.0,
            ppv: 0.0,
            npv: 0.0,
            sensitivity: 0.0,
            specificity: 0.0,
            auroc: 0.0,
            auprc: 0.0,
            mean_pred: 0.0,
            std_pred: 0.0,
            f1_score: 0.0,
            mcc: 0.0,
            balanced_accuracy: 0.0,
        }
    }
}

impl EvaluationMetrics {
    pub fn new(
        loss: f32,
        accuracy: f32,
        ppv: f32,
        npv: f32,
        sensitivity: f32,
        specificity: f32,
        auroc: f32,
        auprc: f32,
        mean_pred: f32,
        std_pred: f32,
        f1_score: f32,
        mcc: f32,
        balanced_accuracy: f32,
    ) -> Self {
        Self {
            loss,
            accuracy,
            ppv,
            npv,
            sensitivity,
            specificity,
            auroc,
            auprc,
            mean_pred,
            std_pred,
            f1_score,
            mcc,
            balanced_accuracy,
        }
    }
}

// For AUROC/AUPRC, the logic is more involved.
// It requires sorting all predictions and calculating the curve.
pub fn calculate_auroc(predictions: &[f32], targets: &[f32]) -> f32 {
    if predictions.is_empty() || targets.is_empty() || predictions.len() != targets.len() {
        return 0.5;
    }

    // Count total positives and negatives
    let total_positives = targets.iter().filter(|&&t| t == 1.0).count();
    let total_negatives = targets.iter().filter(|&&t| t == 0.0).count();

    if total_positives == 0 || total_negatives == 0 {
        return 0.5; // AUROC undefined when only one class present
    }

    // 1. Create pairs of (prediction, target) with indices
    let mut pairs: Vec<(f32, f32)> = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p, t))
        .collect();

    // 2. Sort pairs by prediction descending
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // 3. Iterate through sorted pairs, calculating TPR and FPR at each threshold
    let mut tp = 0u32;
    let mut fp = 0u32;
    let mut prev_tpr = 0.0f32;
    let mut prev_fpr = 0.0f32;
    let mut auroc = 0.0f32;
    let mut prev_pred = f32::INFINITY;

    for (pred, target) in pairs.iter() {
        // When prediction changes, we have a new threshold point
        if *pred != prev_pred && prev_pred != f32::INFINITY {
            let tpr = tp as f32 / total_positives as f32;
            let fpr = fp as f32 / total_negatives as f32;

            // 4. Calculate area using trapezoidal rule
            auroc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;

            prev_tpr = tpr;
            prev_fpr = fpr;
        }

        // Update counts based on actual label
        if *target == 1.0 {
            tp += 1;
        } else {
            fp += 1;
        }

        prev_pred = *pred;
    }

    // Final point (all predictions classified as positive)
    let tpr = tp as f32 / total_positives as f32;
    let fpr = fp as f32 / total_negatives as f32;
    auroc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;

    auroc
}

pub fn calculate_auprc(predictions: &[f32], targets: &[f32]) -> f32 {
    let mut pairs: Vec<(f32, f32)> = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p, t))
        .collect();

    // 2. Sort pairs by prediction descending
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Count total positives and negatives
    let total_positives = targets.iter().filter(|&&t| t == 1.0).count();
    let total_negatives = targets.iter().filter(|&&t| t == 0.0).count();

    // 3. Iterate through sorted pairs, calculating TPR and FPR at each threshold
    let mut tp = 0u32;
    let mut fp = 0u32;
    let mut prev_tpr = 0.0f32;
    let mut prev_fpr = 0.0f32;
    let mut prev_pred = f32::INFINITY;
    let mut auprc = 0.0f32;

    for (pred, target) in pairs.iter() {
        // When prediction changes, we have a new threshold point
        if *pred != prev_pred && prev_pred != f32::INFINITY {
            let tpr = tp as f32 / total_positives as f32;
            let fpr = fp as f32 / total_negatives as f32;

            // 4. Calculate area using trapezoidal rule
            auprc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;

            prev_tpr = tpr;
            prev_fpr = fpr;
        }

        // Update counts based on actual label
        if *target == 1.0 {
            tp += 1;
        } else {
            fp += 1;
        }

        prev_pred = *pred;
    }

    // Final point (all predictions classified as positive)
    let tpr = tp as f32 / total_positives as f32;
    let fpr = fp as f32 / total_negatives as f32;
    auprc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;

    auprc
}

pub fn evalute_auroc<I, O>(
    model: &mut impl Module<I, O>,
    validation_data: &mut MultiModalBatch,
) -> f32 {
    1.0
}
