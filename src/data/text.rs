//! Text data processing module.
//!
//! Provides utilities for loading, tokenizing, and preprocessing
//! text data such as medical notes, patient descriptions, or symptoms.

use crate::data::DataLoader;
use crate::error::AppResult;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

/// Structure for handling text data
pub struct TextData {
    /// Patient identifiers aligned with documents and token_sequences.
    pub patient_ids: Vec<String>,

    /// Original documents
    pub documents: Vec<String>,

    /// Tokenized and converted to word indices
    pub token_sequences: Vec<Vec<usize>>,

    /// Vocabulary: word → index mapping
    pub vocab: HashMap<String, usize>,

    /// Reverse vocabulary: index → word mapping
    pub reverse_vocab: HashMap<usize, String>,

    /// Maximum sequence length
    pub max_seq_length: usize,

    /// Current index for batch iteration
    pub current_idx: usize,
}

impl TextData {
    /// Create a new empty TextData instance
    pub fn new() -> Self {
        TextData {
            patient_ids: Vec::new(),
            documents: Vec::new(),
            token_sequences: Vec::new(),
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            max_seq_length: 0,
            current_idx: 0,
        }
    }

    /// Load text data from files
    ///
    /// # Arguments
    /// * `paths` - Paths to text files, one per document
    /// * `max_vocab_size` - Maximum vocabulary size (most frequent words kept)
    ///
    /// # Returns
    /// Result containing TextData or an error
    pub fn from_files<P: AsRef<Path>>(paths: &[P], max_vocab_size: usize) -> AppResult<Self> {
        let mut documents = Vec::with_capacity(paths.len());

        for path in paths {
            let mut file = File::open(path)?;
            let mut content = String::new();
            file.read_to_string(&mut content)?;
            documents.push(content);
        }

        Self::from_documents(documents, max_vocab_size)
    }

    /// Create TextData from a list of documents
    ///
    /// # Arguments
    /// * `documents` - List of text documents
    /// * `max_vocab_size` - Maximum vocabulary size (most frequent words kept)
    ///
    /// # Returns
    /// Result containing TextData or an error
    pub fn from_documents(documents: Vec<String>, max_vocab_size: usize) -> AppResult<Self> {
        let mut text_data = TextData {
            patient_ids: (0..documents.len()).map(|index| format!("P{:02}", index)).collect(),
            documents: documents.clone(),
            token_sequences: Vec::new(),
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            max_seq_length: 0,
            current_idx: 0,
        };

        // Build vocabulary
        text_data.build_vocabulary(max_vocab_size);

        // Tokenize documents
        text_data.tokenize_documents();

        Ok(text_data)
    }

    /// Build vocabulary from documents
    fn build_vocabulary(&mut self, max_vocab_size: usize) {
        // Count word frequencies
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        for doc in &self.documents {
            for word in Self::tokenize_text(doc) {
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        // Sort by frequency
        let mut word_counts: Vec<(String, usize)> = word_counts.into_iter().collect();
        word_counts.sort_by(|a, b| b.1.cmp(&a.1));

        // Keep only the most frequent words
        let vocab_size = max_vocab_size.min(word_counts.len() + 2); // +2 for <UNK> and <PAD>

        // Add special tokens
        self.vocab.insert("<PAD>".to_string(), 0);
        self.vocab.insert("<UNK>".to_string(), 1);

        self.reverse_vocab.insert(0, "<PAD>".to_string());
        self.reverse_vocab.insert(1, "<UNK>".to_string());

        // Add the most frequent words
        for (i, (word, _)) in word_counts.into_iter().take(vocab_size - 2).enumerate() {
            let idx = i + 2; // +2 for special tokens
            self.vocab.insert(word.clone(), idx);
            self.reverse_vocab.insert(idx, word);
        }
    }

    /// Tokenize all documents using the vocabulary
    fn tokenize_documents(&mut self) {
        self.token_sequences = Vec::with_capacity(self.documents.len());

        for doc in &self.documents {
            let tokens = self.tokenize_document(doc);
            self.max_seq_length = self.max_seq_length.max(tokens.len());
            self.token_sequences.push(tokens);
        }
    }

    /// Split text into tokens (words)
    fn tokenize_text(text: &str) -> Vec<String> {
        // Simple whitespace tokenization with lowercase
        text.split_whitespace()
            .map(|s| s.trim().to_lowercase())
            .map(|s| {
                // Remove punctuation from word boundaries
                let mut chars: Vec<char> = s.chars().collect();

                // Remove punctuation from beginning
                while !chars.is_empty() && chars[0].is_ascii_punctuation() {
                    chars.remove(0);
                }

                // Remove punctuation from end
                while !chars.is_empty() && chars[chars.len() - 1].is_ascii_punctuation() {
                    chars.pop();
                }

                chars.into_iter().collect::<String>()
            })
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Tokenize a document and convert to word indices
    pub fn tokenize_document(&self, text: &str) -> Vec<usize> {
        Self::tokenize_text(text)
            .into_iter()
            .map(|word| {
                *self.vocab.get(&word).unwrap_or(&1) // 1 is <UNK>
            })
            .collect()
    }

    /// Pad sequences to the maximum length
    pub fn pad_sequences(&mut self) {
        for sequence in &mut self.token_sequences {
            let seq_len = sequence.len();
            if seq_len < self.max_seq_length {
                sequence.resize(self.max_seq_length, 0); // 0 is <PAD>
            }
        }
    }

    /// Truncate sequences to the maximum length
    pub fn truncate_sequences(&mut self, max_length: usize) {
        self.max_seq_length = max_length;

        for sequence in &mut self.token_sequences {
            if sequence.len() > max_length {
                sequence.truncate(max_length);
            }
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

impl DataLoader<Vec<Vec<usize>>> for TextData {
    fn next_batch(&mut self, batch_size: usize) -> Option<Vec<Vec<usize>>> {
        if self.current_idx >= self.token_sequences.len() {
            return None;
        }

        let end_idx = (self.current_idx + batch_size).min(self.token_sequences.len());
        let batch = self.token_sequences[self.current_idx..end_idx].to_vec();

        self.current_idx = end_idx;
        Some(batch)
    }

    fn reset(&mut self) {
        self.current_idx = 0;
    }

    fn len(&self) -> usize {
        self.token_sequences.len()
    }
}
