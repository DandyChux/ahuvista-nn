use crate::module::Module;
use rand::Rng;
use std::collections::HashMap;

/// Word embedding layer.
///
/// Maps word indices to dense vector representations.
#[derive(Debug)]
pub struct Embedding {
    /// Vocabulary size
    vocab_size: usize,

    /// Embedding dimension
    embedding_dim: usize,

    /// Embedding weights: vocab_size × embedding_dim
    weights: Vec<f32>,

    /// Accumulated gradients
    grad_weights: Vec<f32>,

    /// Indices used in last forward pass
    last_indices: Vec<usize>,
}

impl Embedding {
    /// Create a new embedding layer.
    ///
    /// # Arguments
    /// * `vocab_size` - Size of vocabulary
    /// * `embedding_dim` - Size of embedding vectors
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize with small random values
        let scale = (1.0 / embedding_dim as f32).sqrt();

        Embedding {
            vocab_size,
            embedding_dim,
            weights: (0..vocab_size * embedding_dim)
                .map(|_| rng.gen_range(-scale..scale))
                .collect(),
            grad_weights: vec![0.0; vocab_size * embedding_dim],
            last_indices: Vec::new(),
        }
    }

    /// Get an embedding vector for a given word index.
    ///
    /// # Arguments
    /// * `idx` - Word index
    ///
    /// # Returns
    /// Embedding vector for the word
    pub fn get_embedding(&self, idx: usize) -> Vec<f32> {
        assert!(idx < self.vocab_size, "Word index out of bounds");

        let start = idx * self.embedding_dim;
        self.weights[start..(start + self.embedding_dim)].to_vec()
    }
}

/// Network for processing text data.
///
/// Combines a word embedding layer with an RNN to process
/// text as sequences of word indices.
#[derive(Debug)]
pub struct TextNet {
    /// Vocabulary map: word → index
    pub vocab: HashMap<String, usize>,

    /// Embedding layer
    embedding: Embedding,

    /// RNN for sequence processing
    rnn_cell: crate::temporal::SimpleRnnCell,

    /// Last hidden state from forward pass
    last_hidden: Vec<f32>,

    /// Cached embedded sequence from forward pass
    cached_embedded_sequence: Vec<Vec<f32>>,
}

impl TextNet {
    /// Create a new text network with a fixed vocabulary.
    ///
    /// # Arguments
    /// * `vocab` - Mapping from words to indices
    /// * `embedding_dim` - Size of word embeddings
    /// * `hidden_size` - Size of RNN hidden state and output vector
    pub fn new(vocab: HashMap<String, usize>, embedding_dim: usize, hidden_size: usize) -> Self {
        let vocab_size = vocab.len();

        TextNet {
            vocab,
            embedding: Embedding::new(vocab_size, embedding_dim),
            rnn_cell: crate::temporal::SimpleRnnCell::new(embedding_dim, hidden_size),
            last_hidden: vec![0.0; hidden_size],
            cached_embedded_sequence: Vec::new(),
        }
    }

    /// Build a vocabulary from a list of texts.
    ///
    /// # Arguments
    /// * `texts` - List of text documents
    /// * `max_vocab_size` - Maximum vocabulary size (most frequent words kept)
    ///
    /// # Returns
    /// Mapping from words to indices
    pub fn build_vocab(texts: &[String], max_vocab_size: usize) -> HashMap<String, usize> {
        // Count word frequencies
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        for text in texts {
            for word in text.split_whitespace() {
                let word = word.to_lowercase();
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        // Sort by frequency
        let mut words: Vec<(String, usize)> = word_counts.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1));

        // Take top words
        let vocab_size = words.len().min(max_vocab_size);

        // Build vocabulary mapping
        let mut vocab = HashMap::with_capacity(vocab_size);
        vocab.insert("<UNK>".to_string(), 0); // Reserve 0 for unknown words

        for (i, (word, _)) in words.into_iter().enumerate().take(vocab_size - 1) {
            vocab.insert(word, i + 1);
        }

        vocab
    }

    /// Convert text to a sequence of word indices.
    ///
    /// # Arguments
    /// * `text` - Input text
    ///
    /// # Returns
    /// Vector of word indices
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| {
                let word = word.to_lowercase();
                *self.vocab.get(&word).unwrap_or(&0) // 0 is <UNK>
            })
            .collect()
    }

    /// Get the output size of this network
    pub fn output_size(&self) -> usize {
        self.last_hidden.len()
    }
}

impl Module<Vec<usize>, Vec<f32>> for TextNet {
    fn forward(&mut self, word_indices: Vec<usize>) -> Vec<f32> {
        if word_indices.is_empty() {
            return vec![0.0; self.rnn_cell.hidden_size];
        }

        // Embed each word
        let embedded_sequence: Vec<Vec<f32>> = word_indices
            .iter()
            .map(|&idx| self.embedding.get_embedding(idx))
            .collect();

        // Cache for backpropagation
        self.embedding.last_indices = word_indices;
        self.cached_embedded_sequence = embedded_sequence.clone();

        // Process through RNN
        self.rnn_cell.reset_state();
        let mut h = vec![0.0; self.rnn_cell.hidden_size];

        // Store initial hidden state
        self.rnn_cell.cached_hiddens.push(h.clone());

        // Process sequence
        for embed in &embedded_sequence {
            h = self.rnn_cell.forward_step(embed, &h);
            self.rnn_cell.cached_hiddens.push(h.clone());
        }

        self.last_hidden = h.clone();
        h
    }

    fn backward(&mut self, grad_output: Vec<f32>) -> Vec<usize> {
        let seq_len = self.cached_embedded_sequence.len();
        if seq_len == 0 {
            return Vec::new();
        }

        // Backpropagate through RNN
        let mut dh_next = grad_output;
        let mut d_embedded = vec![vec![0.0; self.embedding.embedding_dim]; seq_len];

        for t in (0..seq_len).rev() {
            let embed = &self.cached_embedded_sequence[t];
            let h_prev = &self.rnn_cell.cached_hiddens[t];
            let h = &self.rnn_cell.cached_hiddens[t + 1];

            // Gradient through tanh
            let dtanh = crate::temporal::SimpleRnnCell::tanh_derivative(h);
            let dh = dtanh
                .iter()
                .zip(&dh_next)
                .map(|(&dt, &dn)| dt * dn)
                .collect::<Vec<_>>();

            // Accumulate gradients for RNN parameters
            for i in 0..self.rnn_cell.hidden_size {
                // Bias gradient
                self.rnn_cell.grad_b_h[i] += dh[i];

                // Input weights gradient
                for j in 0..self.embedding.embedding_dim {
                    self.rnn_cell.grad_w_ih[i * self.embedding.embedding_dim + j] +=
                        dh[i] * embed[j];
                }

                // Hidden weights gradient
                for j in 0..self.rnn_cell.hidden_size {
                    self.rnn_cell.grad_w_hh[i * self.rnn_cell.hidden_size + j] += dh[i] * h_prev[j];
                }
            }

            // Compute gradient for embedded inputs at this time step
            for j in 0..self.embedding.embedding_dim {
                for i in 0..self.rnn_cell.hidden_size {
                    d_embedded[t][j] +=
                        dh[i] * self.rnn_cell.w_ih[i * self.embedding.embedding_dim + j];
                }
            }

            // Compute gradient for previous hidden state
            let mut dh_prev = vec![0.0; self.rnn_cell.hidden_size];
            for j in 0..self.rnn_cell.hidden_size {
                for i in 0..self.rnn_cell.hidden_size {
                    dh_prev[j] += dh[i] * self.rnn_cell.w_hh[i * self.rnn_cell.hidden_size + j];
                }
            }

            // Prepare for next iteration
            dh_next = dh_prev;
        }

        // Backpropagate through embedding
        for t in 0..seq_len {
            let idx = self.embedding.last_indices[t];
            let embed_start = idx * self.embedding.embedding_dim;

            for j in 0..self.embedding.embedding_dim {
                self.embedding.grad_weights[embed_start + j] += d_embedded[t][j];
            }
        }

        // We can't return actual gradients for discrete indices,
        // so return the original indices
        self.embedding.last_indices.clone()
    }

    fn update(&mut self, lr: f32) {
        // Update embedding weights
        for i in 0..self.embedding.weights.len() {
            self.embedding.weights[i] -= lr * self.embedding.grad_weights[i];
        }

        // Update RNN parameters
        // Input weights
        for i in 0..self.rnn_cell.w_ih.len() {
            self.rnn_cell.w_ih[i] -= lr * self.rnn_cell.grad_w_ih[i];
        }

        // Hidden weights
        for i in 0..self.rnn_cell.w_hh.len() {
            self.rnn_cell.w_hh[i] -= lr * self.rnn_cell.grad_w_hh[i];
        }

        // Biases
        for i in 0..self.rnn_cell.b_h.len() {
            self.rnn_cell.b_h[i] -= lr * self.rnn_cell.grad_b_h[i];
        }
    }

    fn zero_grad(&mut self) {
        // Zero embedding gradients
        for grad in &mut self.embedding.grad_weights {
            *grad = 0.0;
        }

        // Zero RNN gradients
        for grad in &mut self.rnn_cell.grad_w_ih {
            *grad = 0.0;
        }

        for grad in &mut self.rnn_cell.grad_w_hh {
            *grad = 0.0;
        }

        for grad in &mut self.rnn_cell.grad_b_h {
            *grad = 0.0;
        }
    }
}
