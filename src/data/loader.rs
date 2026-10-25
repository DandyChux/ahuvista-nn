//! Prepared model data and mini-batch loading.
//!
//! This module does not parse CSV/FHIR input and does not access SQLite.
//! The database layer creates patient-aligned PreparedRecords. This module
//! validates, splits, and batches them.

use std::collections::HashMap;

use ndarray::Array3;
use rand::seq::SliceRandom;

use crate::{
    error::{AppError, AppResult},
    models::PatientSequence,
};

#[derive(Debug, Clone)]
pub struct PreparedRecord {
    pub patient_id: String,
    pub sequence: PatientSequence,
    pub text: Vec<usize>,
    pub images: Vec<Array3<f32>>,
    pub target: f32,
    pub cause_code: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct PreparedDataset {
    pub records: Vec<PreparedRecord>,
    pub text_vocabulary: HashMap<String, usize>,
    pub image_channels: Option<usize>,
    pub image_size: Option<(usize, usize)>
}

impl PreparedDataset {
    pub fn from_records(
        records: Vec<PreparedRecord>,
        text_vocabulary: HashMap<String, usize>,
        image_channels: Option<usize>,
        image_size: Option<(usize, usize)>
    ) -> AppResult<Self> {
        if records.is_empty() {
            return Err(AppError::InvalidData(
                "Prepared dataset is empty".to_string(),
            ));
        }

        validate_records(&records)?;

        Ok(Self {
            records,
            text_vocabulary,
            image_channels,
            image_size,
        })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn has_text(&self) -> bool {
        self.records
            .iter()
            .any(|record| !record.text.is_empty())
    }

    pub fn has_images(&self) -> bool {
        self.records
            .iter()
            .any(|record| !record.images.is_empty())
    }

    pub fn sequences(&self) -> Vec<PatientSequence> {
        self.records
            .iter()
            .map(|record| record.sequence.clone())
            .collect()
    }

    pub fn targets(&self) -> Vec<f32> {
        self.records
            .iter()
            .map(|record| record.target)
            .collect()
    }

    pub fn causes(&self) -> Vec<usize> {
        self.records
            .iter()
            .map(|record| {
                record
                    .cause_code
                    .map(|code| code as usize)
                    .unwrap_or(0)
            })
            .collect()
    }

    pub fn split_at(
        &self,
        index: usize,
    ) -> AppResult<(Self, Self)> {
        if index > self.len() {
            return Err(AppError::InvalidData(format!(
                "Split index {} exceeds dataset length {}",
                index,
                self.len()
            )));
        }

        let train =
            Self::from_records(
                self.records[..index].to_vec(),
                self.text_vocabulary.clone(),
                self.image_channels,
                self.image_size,
            )?;

        let validation =
            Self::from_records(
                self.records[index..].to_vec(),
                self.text_vocabulary.clone(),
                self.image_channels,
                self.image_size,
            )?;

        Ok((train, validation))
    }

    pub fn split(
        &self,
        train_ratio: f32,
    ) -> AppResult<(Self, Self)> {
        if !(0.0..=1.0).contains(&train_ratio) {
            return Err(AppError::InvalidData(format!(
                "Train ratio must be between 0 and 1, got {}",
                train_ratio
            )));
        }

        let split_index =
            (self.len() as f32 * train_ratio).floor() as usize;

        if split_index > self.len() {
        	return Err(AppError::InvalidData(format!(
        		"Split index {} is greater than dataset size {}",
        		split_index, self.len()
        	)));
        }

        self.split_at(split_index)
    }

    pub fn require_images(
        &self,
    ) -> AppResult<()> {
    	let missing = self
	        .records
			.iter()
			.filter(|record| record.images.is_empty())
			.count();

        if missing > 0 {
            return Err(AppError::InvalidData(format!(
                "Image modality is enabled but {} patients have no images", missing
            )));
        }

        Ok(())
    }
}

fn validate_records(
    records: &[PreparedRecord],
) -> AppResult<()> {
    let static_width =
        records[0].sequence.static_features.len();

    let dynamic_width =
        records
            .iter()
            .flat_map(|record| {
                record.sequence.dynamic_features.iter()
            })
            .next()
            .map(Vec::len);

    for record in records {
        if record.sequence.static_features.len()
            != static_width
        {
            return Err(AppError::InvalidData(format!(
                "Static feature width differs for patient {}",
                record.patient_id
            )));
        }

        for timestep in &record.sequence.dynamic_features {
            if Some(timestep.len()) != dynamic_width {
                return Err(AppError::InvalidData(format!(
                    "Dynamic feature width differs for patient {}",
                    record.patient_id
                )));
            }
        }

        if !record.target.is_finite() {
            return Err(AppError::InvalidData(format!(
                "Target for patient {} is not finite",
                record.patient_id
            )));
        }

        if record.images.iter().flatten().any(|value| {
            !value.is_finite()
        }) {
            return Err(AppError::InvalidData(format!(
                "Image for patient {} contains a non-finite value",
                record.patient_id
            )));
        }
    }

    Ok(())
}

pub trait DataLoader<T> {
    fn next_batch(&mut self, batch_size: usize) -> Option<T>;
    fn reset(&mut self);
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub struct MultiModalBatch {
    pub patient_ids: Vec<String>,
    pub sequences: Vec<PatientSequence>,
    pub text: Vec<Vec<usize>>,
    pub images: Vec<Vec<Array3<f32>>>,
    pub targets: Vec<f32>,
    pub causes: Vec<usize>,
    pub indices: Vec<usize>
}

impl MultiModalBatch {
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct MultiModalDataLoader {
    dataset: PreparedDataset,
    current_idx: usize,
    indices: Vec<usize>,
}

impl MultiModalDataLoader {
    pub fn new(dataset: PreparedDataset) -> Self {
        let len = dataset.len();

        Self {
            dataset,
            current_idx: 0,
            indices: (0..len).collect(),
        }
    }

    pub fn dataset(&self) -> &PreparedDataset {
        &self.dataset
    }

    pub fn dataset_mut(&mut self) -> &mut PreparedDataset {
        &mut self.dataset
    }

    pub fn shuffle(&mut self) {
        let mut rng = rand::rng();
        self.indices.shuffle(&mut rng);
        self.current_idx = 0;
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

impl DataLoader<MultiModalBatch>
    for MultiModalDataLoader
{
	fn next_batch(
    &mut self,
    batch_size: usize,
	) -> Option<MultiModalBatch> {
	    if batch_size == 0
	        || self.current_idx >= self.dataset.len()
	    {
	        return None;
	    }

	    let end =
	        (self.current_idx + batch_size)
	            .min(self.dataset.len());

	    let selected =
	        &self.indices[self.current_idx..end];

	    let include_text =
	        self.dataset.has_text();

	    let include_images =
	        self.dataset.has_images();

	    let mut patient_ids =
	        Vec::with_capacity(selected.len());

	    let mut sequences =
	        Vec::with_capacity(selected.len());

	    let mut text =
	        if include_text {
	            Vec::with_capacity(selected.len())
	        } else {
	            Vec::new()
	        };

	    let mut images =
	        if include_images {
	            Vec::with_capacity(selected.len())
	        } else {
	            Vec::new()
	        };

	    let mut targets =
	        Vec::with_capacity(selected.len());

	    let mut causes =
	        Vec::with_capacity(selected.len());

	    for &dataset_index in selected {
	        let record =
	            &self.dataset.records[dataset_index];

	        patient_ids.push(
	            record.patient_id.clone()
	        );

	        sequences.push(
	            record.sequence.clone()
	        );

	        if include_text {
	            text.push(
	                record.text.clone()
	            );
	        }

	        if include_images {
	            images.push(
	                record.images.clone()
	            );
	        }

	        targets.push(record.target);

	        causes.push(
	            record
	                .cause_code
	                .map(|code| code as usize)
	                .unwrap_or(0)
	        );
	    }

	    let indices =
	        selected.to_vec();

	    self.current_idx =
	        end;

	    Some(MultiModalBatch {
	        patient_ids,
	        sequences,
	        text,
	        images,
	        targets,
	        causes,
	        indices,
	    })
	}

    fn reset(&mut self) {
        self.current_idx = 0;
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

pub fn create_data_loader(
    dataset: PreparedDataset,
) -> MultiModalDataLoader {
    MultiModalDataLoader::new(dataset)
}
