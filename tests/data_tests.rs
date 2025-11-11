//! Comprehensive tests for data loading and preprocessing

use ahuvista_nn::data::{
    preprocessing::*, DataLoader, ImageData, TabularData, TemporalData, TextData,
};
use ndarray::Array3;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

#[cfg(test)]
mod tabular_tests {
    use super::*;

    fn create_test_csv(path: &str, with_header: bool) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        if with_header {
            writeln!(file, "age,height,weight,gender")?;
        }
        writeln!(file, "25,170.5,65.2,female")?;
        writeln!(file, "30,180.0,75.5,male")?;
        writeln!(file, "22,165.3,58.0,female")?;
        writeln!(file, "35,175.8,70.1,male")?;
        writeln!(file, "28,168.2,62.5,female")?;

        Ok(())
    }

    #[test]
    fn test_tabular_data_creation() {
        let data = TabularData::new();
        assert_eq!(data.data.len(), 0);
        assert_eq!(data.feature_names.len(), 0);
    }

    #[test]
    fn test_tabular_data_from_csv_with_header() {
        let test_file = "tests/data/tabular/part_1.csv";
        fs::create_dir_all("tests/data/tabular").ok();
        create_test_csv(test_file, true).unwrap();

        let categorical = vec!["gender".to_string()];
        let data = TabularData::from_csv(test_file, true, &categorical).unwrap();

        assert_eq!(data.data.len(), 5);
        assert_eq!(data.feature_names.len(), 4);
        assert!(data.categorical_features.contains(&"gender".to_string()));

        // Clean up
        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_tabular_data_from_csv_without_header() {
        let test_file = "tests/data/tabular/test_no_header.csv";
        fs::create_dir_all("tests/data/tabular").ok();
        create_test_csv(test_file, false).unwrap();

        let data = TabularData::from_csv(test_file, false, &[]).unwrap();

        assert_eq!(data.data.len(), 5);
        assert!(data.feature_names[0].starts_with("Feature"));

        // Clean up
        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_tabular_data_normalization() {
        let test_file = "tests/data/tabular/test_normalize.csv";
        fs::create_dir_all("tests/data/tabular").ok();
        create_test_csv(test_file, true).unwrap();

        let data = TabularData::from_csv(test_file, true, &["gender".to_string()]).unwrap();

        // Check that numeric features are normalized
        for row in &data.data {
            for (i, &value) in row.iter().enumerate() {
                let feature_name = &data.feature_names[i];
                if !data.categorical_features.contains(feature_name) {
                    // Normalized values should typically be in range [-3, 3] for most data
                    assert!(value > -10.0 && value < 10.0);
                }
            }
        }

        // Clean up
        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_tabular_data_feature_dim() {
        let test_file = "tests/data/tabular/test_dim.csv";
        fs::create_dir_all("tests/data/tabular").ok();
        create_test_csv(test_file, true).unwrap();

        let data = TabularData::from_csv(test_file, true, &[]).unwrap();
        assert_eq!(data.feature_dim(), 4);

        // Clean up
        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_tabular_data_loader() {
        let test_file = "tests/data/tabular/part_1.csv";
        fs::create_dir_all("tests/data/tabular").ok();
        create_test_csv(test_file, true).unwrap();

        let mut data = TabularData::from_csv(test_file, true, &[]).unwrap();

        // Test batch loading
        let batch1 = data.next_batch(2);
        assert!(batch1.is_some());
        assert_eq!(batch1.unwrap().len(), 2);

        let batch2 = data.next_batch(2);
        assert!(batch2.is_some());
        assert_eq!(batch2.unwrap().len(), 2);

        let batch3 = data.next_batch(2);
        assert!(batch3.is_some());
        assert_eq!(batch3.unwrap().len(), 1); // Only 1 sample left

        let batch4 = data.next_batch(2);
        assert!(batch4.is_none()); // No more data

        // Test reset
        data.reset();
        let batch_after_reset = data.next_batch(5);
        assert!(batch_after_reset.is_some());
        assert_eq!(batch_after_reset.unwrap().len(), 5);

        // Clean up
        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_tabular_data_preprocess_row() {
        let test_file = "tests/data/tabular/test_preprocess.csv";
        fs::create_dir_all("tests/data/tabular").ok();
        create_test_csv(test_file, true).unwrap();

        let data = TabularData::from_csv(test_file, true, &["gender".to_string()]).unwrap();

        let new_row = vec![
            "27".to_string(),
            "172.0".to_string(),
            "68.5".to_string(),
            "male".to_string(),
        ];

        let processed = data.preprocess_row(&new_row);
        assert_eq!(processed.len(), 4);

        // Clean up
        fs::remove_file(test_file).ok();
    }
}

#[cfg(test)]
mod text_tests {
    use super::*;

    #[test]
    fn test_text_data_creation() {
        let data = TextData::new();
        assert_eq!(data.documents.len(), 0);
        assert_eq!(data.token_sequences.len(), 0);
    }

    #[test]
    fn test_text_data_from_documents() {
        let documents = vec![
            "Hello world this is a test".to_string(),
            "Another test document here".to_string(),
            "Machine learning is great".to_string(),
        ];

        let data = TextData::from_documents(documents.clone(), 100).unwrap();

        assert_eq!(data.documents.len(), 3);
        assert_eq!(data.token_sequences.len(), 3);
        assert!(data.vocab.len() > 0);
    }

    #[test]
    fn test_text_data_vocabulary() {
        let documents = vec!["test test test".to_string(), "hello world".to_string()];

        let data = TextData::from_documents(documents, 100).unwrap();

        // Should have special tokens plus regular words
        assert!(data.vocab.contains_key("<PAD>"));
        assert!(data.vocab.contains_key("<UNK>"));
        assert!(data.vocab.contains_key("test"));
    }

    #[test]
    fn test_text_data_tokenization() {
        let documents = vec!["Hello, World! This is a test.".to_string()];

        let data = TextData::from_documents(documents, 100).unwrap();

        // Verify tokenization happened
        assert!(data.token_sequences.len() > 0);
        assert!(data.token_sequences[0].len() > 0);
    }

    #[test]
    fn test_text_data_padding() {
        let documents = vec![
            "short".to_string(),
            "this is a much longer sentence".to_string(),
        ];

        let mut data = TextData::from_documents(documents, 100).unwrap();

        let max_len_before = data.max_seq_length;
        data.pad_sequences();

        // All sequences should now have the same length
        assert!(data
            .token_sequences
            .iter()
            .all(|seq| seq.len() == max_len_before));
    }

    #[test]
    fn test_text_data_truncation() {
        let documents = vec!["this is a very long sentence with many words".to_string()];

        let mut data = TextData::from_documents(documents, 100).unwrap();

        data.truncate_sequences(5);

        assert_eq!(data.max_seq_length, 5);
        assert!(data.token_sequences[0].len() <= 5);
    }

    #[test]
    fn test_text_data_loader() {
        let documents = vec![
            "doc one".to_string(),
            "doc two".to_string(),
            "doc three".to_string(),
            "doc four".to_string(),
        ];

        let mut data = TextData::from_documents(documents, 100).unwrap();

        let batch1 = data.next_batch(2);
        assert!(batch1.is_some());
        assert_eq!(batch1.unwrap().len(), 2);

        let batch2 = data.next_batch(3);
        assert!(batch2.is_some());
        assert_eq!(batch2.unwrap().len(), 2); // Only 2 left

        data.reset();
        let batch3 = data.next_batch(4);
        assert!(batch3.is_some());
        assert_eq!(batch3.unwrap().len(), 4);
    }

    #[test]
    fn test_text_data_vocab_size_limit() {
        let documents = vec![
            "word1 word2 word3 word4 word5".to_string(),
            "word6 word7 word8 word9 word10".to_string(),
        ];

        let data = TextData::from_documents(documents, 5).unwrap();

        // Vocab size should be limited (5 + 2 special tokens = 7 max)
        assert!(data.vocab.len() <= 7);
    }
}

#[cfg(test)]
mod temporal_tests {
    use super::*;

    fn create_test_temporal_csv(path: &str) -> std::io::Result<()> {
        fs::create_dir_all(std::path::Path::new(path).parent().unwrap())?;
        let mut file = File::create(path)?;

        writeln!(file, "time,temperature,heart_rate,blood_pressure")?;
        writeln!(file, "0,36.5,70,120")?;
        writeln!(file, "1,36.7,72,118")?;
        writeln!(file, "2,36.8,75,122")?;

        Ok(())
    }

    #[test]
    fn test_temporal_data_creation() {
        let data = TemporalData::new();
        assert_eq!(data.sequences.len(), 0);
        assert_eq!(data.max_seq_length, 0);
    }

    #[test]
    fn test_temporal_data_from_single_csv() {
        let test_file = "tests/data/temporal/part_1.csv";
        create_test_temporal_csv(test_file).unwrap();

        let data = TemporalData::from_single_csv(
            test_file,
            true,
            "patient_id".to_string(),
            Some("timestamp".to_string()),
        )
        .unwrap();

        assert_eq!(data.sequences.len(), 1);
        assert!(data.sequences[0].len() > 0);

        // Clean up
        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_temporal_data_feature_dim() {
        let test_file = "tests/data/temporal/test_dim.csv";
        create_test_temporal_csv(test_file).unwrap();

        let data = TemporalData::from_single_csv(
            test_file,
            true,
            "patient_id".to_string(),
            Some("timestamp".to_string()),
        )
        .unwrap();

        // Should have 3 features (excluding time column)
        assert_eq!(data.feature_dim(), 3);

        // Clean up
        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_temporal_data_padding() {
        let test_file1 = "tests/data/temporal/test_pad1.csv";
        let test_file2 = "tests/data/temporal/test_pad2.csv";

        create_test_temporal_csv(test_file1).unwrap();

        // Create shorter sequence
        fs::create_dir_all("tests/data/temporal").ok();
        let mut file2 = File::create(test_file2).unwrap();
        writeln!(file2, "time,temperature,heart_rate,blood_pressure").unwrap();
        writeln!(file2, "0,36.5,70,120").unwrap();

        let paths = vec![test_file1, test_file2];
        let mut data =
            TemporalData::from_csv_files(&paths, true, Some("time".to_string())).unwrap();

        let max_len = data.max_seq_length;
        data.pad_sequences();

        // All sequences should have the same length
        assert!(data.sequences.iter().all(|seq| seq.len() == max_len));

        // Clean up
        fs::remove_file(test_file1).ok();
        fs::remove_file(test_file2).ok();
    }

    #[test]
    fn test_temporal_data_loader() {
        let test_file1 = "tests/data/temporal/test_load1.csv";
        let test_file2 = "tests/data/temporal/test_load2.csv";
        let test_file3 = "tests/data/temporal/test_load3.csv";

        create_test_temporal_csv(test_file1).unwrap();
        create_test_temporal_csv(test_file2).unwrap();
        create_test_temporal_csv(test_file3).unwrap();

        let paths = vec![test_file1, test_file2, test_file3];
        let mut data = TemporalData::from_csv_files(&paths, true, None).unwrap();

        let batch1 = data.next_batch(2);
        assert!(batch1.is_some());
        assert_eq!(batch1.unwrap().len(), 2);

        data.reset();
        let batch2 = data.next_batch(5);
        assert!(batch2.is_some());
        assert_eq!(batch2.unwrap().len(), 3);

        // Clean up
        fs::remove_file(test_file1).ok();
        fs::remove_file(test_file2).ok();
        fs::remove_file(test_file3).ok();
    }
}

#[cfg(test)]
mod image_tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn create_test_image(path: &str, width: u32, height: u32) -> std::io::Result<()> {
        fs::create_dir_all(std::path::Path::new(path).parent().unwrap())?;

        let img = ImageBuffer::from_fn(width, height, |x, y| {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = 128u8;
            Rgb([r, g, b])
        });

        img.save(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    #[test]
    fn test_image_data_creation() {
        let data = ImageData::new();
        assert_eq!(data.images.len(), 0);
        assert_eq!(data.channels, 0);
    }

    #[test]
    fn test_image_data_from_files_rgb() {
        let test_img1 = "tests/data/images/test1.png";
        let test_img2 = "tests/data/images/test2.png";

        create_test_image(test_img1, 64, 64).unwrap();
        create_test_image(test_img2, 64, 64).unwrap();

        let paths = vec![test_img1, test_img2];
        let data = ImageData::from_files(&paths, (32, 32), false).unwrap();

        assert_eq!(data.images.len(), 2);
        assert_eq!(data.channels, 3); // RGB
        assert_eq!(data.image_size, (32, 32));
        assert_eq!(data.channel_means.len(), 3);
        assert_eq!(data.channel_stds.len(), 3);

        // Clean up
        fs::remove_file(test_img1).ok();
        fs::remove_file(test_img2).ok();
    }

    #[test]
    fn test_image_data_from_files_grayscale() {
        let test_img = "tests/data/images/test_gray.png";
        create_test_image(test_img, 64, 64).unwrap();

        let paths = vec![test_img];
        let data = ImageData::from_files(&paths, (32, 32), true).unwrap();

        assert_eq!(data.channels, 1); // Grayscale
        assert_eq!(data.channel_means.len(), 1);

        // Clean up
        fs::remove_file(test_img).ok();
    }

    #[test]
    fn test_image_data_normalization() {
        let test_img = "tests/data/images/test_norm.png";
        create_test_image(test_img, 64, 64).unwrap();

        let paths = vec![test_img];
        let data = ImageData::from_files(&paths, (32, 32), false).unwrap();

        // Check that images are normalized (mean ≈ 0, std ≈ 1)
        for img in &data.images {
            for c in 0..data.channels {
                let channel_data: Vec<f32> =
                    img.slice(ndarray::s![c, .., ..]).iter().cloned().collect();
                let mean: f32 = channel_data.iter().sum::<f32>() / channel_data.len() as f32;

                // Normalized data should have mean close to 0
                assert!(mean.abs() < 0.5);
            }
        }

        // Clean up
        fs::remove_file(test_img).ok();
    }

    #[test]
    fn test_image_data_loader() {
        let test_img1 = "tests/data/images/test_batch1.png";
        let test_img2 = "tests/data/images/test_batch2.png";
        let test_img3 = "tests/data/images/test_batch3.png";

        create_test_image(test_img1, 64, 64).unwrap();
        create_test_image(test_img2, 64, 64).unwrap();
        create_test_image(test_img3, 64, 64).unwrap();

        let paths = vec![test_img1, test_img2, test_img3];
        let mut data = ImageData::from_files(&paths, (32, 32), false).unwrap();

        let batch1 = data.next_batch(2);
        assert!(batch1.is_some());
        assert_eq!(batch1.unwrap().len(), 2);

        let batch2 = data.next_batch(2);
        assert!(batch2.is_some());
        assert_eq!(batch2.unwrap().len(), 1);

        data.reset();
        let batch3 = data.next_batch(3);
        assert!(batch3.is_some());
        assert_eq!(batch3.unwrap().len(), 3);

        // Clean up
        fs::remove_file(test_img1).ok();
        fs::remove_file(test_img2).ok();
        fs::remove_file(test_img3).ok();
    }

    #[test]
    fn test_image_data_preprocessing() {
        let test_img = "tests/data/images/test_preprocess.png";
        create_test_image(test_img, 64, 64).unwrap();

        let paths = vec![test_img];
        let data = ImageData::from_files(&paths, (32, 32), false).unwrap();

        // Preprocess a new image
        let test_img2 = "tests/data/images/test_preprocess2.png";
        create_test_image(test_img2, 128, 128).unwrap();

        let preprocessed = data.preprocess_image(test_img2).unwrap();

        assert_eq!(preprocessed.shape(), &[3, 32, 32]);

        // Clean up
        fs::remove_file(test_img).ok();
        fs::remove_file(test_img2).ok();
    }
}

#[cfg(test)]
mod preprocessing_tests {
    use super::*;

    #[test]
    fn test_split_data_with_weights() {
        let tabular_data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];

        let temporal_data = vec![
            vec![vec![1.0], vec![2.0]],
            vec![vec![3.0], vec![4.0]],
            vec![vec![5.0], vec![6.0]],
            vec![vec![7.0], vec![8.0]],
        ];

        let text_data = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12],
        ];

        let image_data = vec![
            Array3::zeros((3, 32, 32)),
            Array3::zeros((3, 32, 32)),
            Array3::zeros((3, 32, 32)),
            Array3::zeros((3, 32, 32)),
        ];

        let targets = vec![0.0, 1.0, 0.0, 1.0];
        let weights = vec![1.0, 1.5, 1.0, 1.2];

        let (train, val, train_targets, val_targets, train_weights, val_weights) =
            split_data_with_weights(
                &tabular_data,
                &temporal_data,
                &text_data,
                &image_data,
                &targets,
                &weights,
                0.75,
            );

        assert_eq!(train.0.len(), 3); // tabular
        assert_eq!(val.0.len(), 1);
        assert_eq!(train_targets.len(), 3);
        assert_eq!(val_targets.len(), 1);
        assert_eq!(train_weights.len(), 3);
        assert_eq!(val_weights.len(), 1);
    }
}
