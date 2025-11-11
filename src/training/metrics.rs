use crate::{core::Module, data::MultiModalBatch};

pub struct ConfusionMatrix {
    pub tp: u32,  // True Positives
    pub fp: u32,  // False Positives
    pub tn: u32,  // True Negatives
    pub fn_: u32, // False Negatives
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
    pub loss: f32,
    pub accuracy: f32,
    pub ppv: f32,
    pub npv: f32,
    pub sensitivity: f32, // Also known as Recall or True Positive Rate
    pub specificity: f32, // True Negative Rate
    pub auroc: f32,
    // pub auprc: f32,             // Area Under Precision-Recall Curve
    // pub f1_score: f32,          // Harmonic mean of precision and recall
    // pub mcc: f32,               // Matthews Correlation Coefficient
    // pub balanced_accuracy: f32, // Average of sensitivity and specificity
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
        // auprc: f32,
        // f1_score: f32,
        // mcc: f32,
        // balanced_accuracy: f32,
    ) -> Self {
        Self {
            loss,
            accuracy,
            ppv,
            npv,
            sensitivity,
            specificity,
            auroc,
        }
    }
}

// For AUROC/AUPRC, the logic is more involved.
// It requires sorting all predictions and calculating the curve.
pub fn calculate_auroc(predictions: &[f32], targets: &[f32]) -> f32 {
    // 1. Create pairs of (prediction, target)
    // 2. Sort pairs by prediction descending
    // 3. Iterate through sorted pairs, updating TPR and FPR to trace the curve
    // 4. Calculate area using trapezoidal rule
    // This is a placeholder for the full implementation.
    0.5 // a random classifier's score
}

pub fn evalute_auroc<I, O>(
    model: &mut impl Module<I, O>,
    validation_data: &mut MultiModalBatch,
) -> f32 {
    1.0
}
