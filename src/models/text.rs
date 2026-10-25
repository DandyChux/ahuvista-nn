use crate::core::Module;
use rand::Rng;
use std::collections::HashMap;

/// Word embedding layer.
///
/// Maps word indices to dense vector representations.
#[derive(Debug)]
pub struct EmbeddingLayer {
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

impl EmbeddingLayer {
    /// Create a new embedding layer.
    ///
    /// # Arguments
    /// * `vocab_size` - Size of vocabulary
    /// * `embedding_dim` - Size of embedding vectors
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::rng();

        // Initialize with small random values
        let scale = (1.0 / embedding_dim as f32).sqrt();

        EmbeddingLayer {
            vocab_size,
            embedding_dim,
            weights: (0..vocab_size * embedding_dim)
                .map(|_| rng.random_range(-scale..scale))
                .collect(),
            grad_weights: vec![0.0; vocab_size * embedding_dim],
            last_indices: Vec::new(),
        }
    }

    /// Get all parameters from the embedding layer.
    pub fn get_parameters(&self) -> Vec<f32> {
        self.weights.clone()
    }

    /// Set all parameters in the embedding layer.
    /// Returns the number of parameters consumed.
    pub fn set_parameters(&mut self, params: &[f32]) -> Result<usize, Box<dyn std::error::Error>> {
        let weights_len = self.weights.len();
        if params.len() < weights_len {
            return Err(format!(
                "Not enough parameters. Expected {}, got {}",
                weights_len,
                params.len()
            )
            .into());
        }
        self.weights.copy_from_slice(&params[..weights_len]);
        Ok(weights_len)
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
    embedding: EmbeddingLayer,

    /// RNN for sequence processing
    rnn_cell: crate::models::SimpleRnnCell,

    hidden_size: usize,

    /// Last hidden state from forward pass
    last_hidden: Vec<f32>,

    cached_tokens: Vec<usize>,

    /// Cached embedded sequence from forward pass
    cached_embedded_sequence: Vec<Vec<f32>>,

    has_cached_forward: bool,
}

impl TextNet {
    /// Create a new text network with a fixed vocabulary.
    ///
    /// # Arguments
    /// * `vocab` - Mapping from words to indices
    /// * `embedding_dim` - Size of word embeddings
    /// * `hidden_size` - Size of RNN hidden state and output vector
    pub fn new(vocab: HashMap<String, usize>, embedding_dim: usize, hidden_size: usize) -> Self {
        let vocab_size = vocab.len().max(1);

        TextNet {
            vocab,
            embedding: EmbeddingLayer::new(vocab_size, embedding_dim),
            rnn_cell: crate::models::SimpleRnnCell::new(embedding_dim, hidden_size),
            hidden_size,
            last_hidden: vec![0.0; hidden_size],
            cached_tokens: Vec::new(),
            cached_embedded_sequence: Vec::new(),
            has_cached_forward: false,
        }
    }

    pub fn output_size(&self) -> usize {
        self.hidden_size
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding.embedding_dim
    }

    pub fn vocab_size(&self) -> usize {
        self.embedding.vocab_size
    }

    /// Get all parameters from the network.
    pub fn get_parameters(&self) -> Vec<f32> {
        let mut params = self.embedding.get_parameters();
        params.extend(self.rnn_cell.get_parameters());
        params
    }

    /// Set all parameters in the network.
    /// Returns the number of parameters consumed.
    pub fn set_parameters(&mut self, params: &[f32]) -> Result<usize, Box<dyn std::error::Error>> {
        let mut offset = 0;

        offset += self.embedding.set_parameters(&params[offset..])?;
        offset += self.rnn_cell.set_parameters(&params[offset..])?;

        Ok(offset)
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
        let mut counts: HashMap<String, usize> = HashMap::new();

        for text in texts {
            for word in text.split_whitespace() {
                let word = word.to_lowercase();

                *counts.entry(word).or_insert(0) += 1;
            }
        }

        // Sort by frequency
        let mut words: Vec<(String, usize)> = counts.into_iter().collect();

        words.sort_by(|left, right| {
            right
            	.1
             	.cmp(&left.1)
             	.then_with(|| left.0.cmp(&right.0))
        });

        // Take top words
        let max_vocab_size = max_vocab_size.max(2);

        // Build vocabulary mapping
        let mut vocab = HashMap::with_capacity(max_vocab_size);

        vocab.insert("<UNK>".to_string(), 0); // Reserve 0 for unknown words

        for (i, (word, _)) in words
        	.into_iter()
         	.take(max_vocab_size - 1)
         	.enumerate()
        {
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
                let normalized = word.to_lowercase();

                self.vocab.get(&normalized).copied().unwrap_or(0)
            })
            .collect()
    }

    fn clear_forward_cache(&mut self) {
    	self.cached_tokens.clear();
     	self.cached_embedded_sequence.clear();
      	self.last_hidden.fill(0.0);
       	self.has_cached_forward = false;
    }
}

impl Module<Vec<usize>, Vec<f32>> for TextNet {
    fn forward(&mut self, input: Vec<usize>) -> Vec<f32> {
	    self.clear_forward_cache();

    	if input.is_empty() {
     		self.cached_tokens = Vec::new();
	       	self.has_cached_forward = true;

            return vec![0.0; self.hidden_size];
        }

        for &index in &input {
        	assert!(
         		index < self.embedding.vocab_size,
           		"Token index {} is outside vocab size {}",
             	index,
              	self.embedding.vocab_size
         	);
        }

        // Embed each word
        let embedded: Vec<Vec<f32>> = input
            .iter()
            .map(|&idx| self.embedding.get_embedding(idx))
            .collect();

        // Cache for backpropagation
        self.cached_tokens = input.clone();
        self.embedding.last_indices = input;
        self.cached_embedded_sequence = embedded.clone();

        // Process through RNN
        self.rnn_cell.reset_state();
        let mut hidden = vec![0.0; self.rnn_cell.hidden_size];

        // Store initial hidden state
        self.rnn_cell.cached_hiddens.clear();
        self.rnn_cell.cached_hiddens.push(hidden.clone());

        // Process sequence
        for embedding in &embedded {
            hidden = self.rnn_cell.forward_step(embedding, &hidden);
            self.rnn_cell.cached_hiddens.push(hidden.clone());
        }

        self.last_hidden = hidden.clone();
        self.has_cached_forward = true;

        hidden
    }

    fn backward(&mut self, grad_output: Vec<f32>) -> Vec<usize> {
	   	if !self.has_cached_forward {
			return Vec::new();
		}

		// There are no token parameters to differntiate for an empty sequence,
		// so the gradient is empty
		if self.cached_tokens.is_empty() {
			return Vec::new();
		}

		assert_eq!(
			grad_output.len(),
			self.hidden_size,
			"Text gradient size mismatch: expected {}, got {}",
			self.hidden_size,
			grad_output.len()
		);

		let sequence_length = self.cached_embedded_sequence.len();

		if sequence_length == 0 {
			return Vec::new();
		}

        // Backpropagate through RNN
        let mut dh_next = grad_output;
        let mut embedded_gradients = vec![vec![0.0; self.embedding.embedding_dim]; sequence_length];

        for time in (0..sequence_length).rev() {
            let embedding = &self.cached_embedded_sequence[time];
            let previous_hidden = &self.rnn_cell.cached_hiddens[time];
            let hidden = &self.rnn_cell.cached_hiddens[time + 1];

            // Gradient through tanh
            let tanh_derivative = crate::models::SimpleRnnCell::tanh_derivative(hidden);
            let hidden_gradient = tanh_derivative
                .iter()
                .zip(&dh_next)
                .map(|(&derivative, &gradient)| derivative * gradient)
                .collect::<Vec<_>>();

            // Accumulate gradients for RNN parameters
            for hidden_index in 0..self.rnn_cell.hidden_size {
                // Bias gradient
                self.rnn_cell.grad_b_h[hidden_index] += hidden_gradient[hidden_index];

                // Input weights gradient
                for embedding_index in 0..self.embedding.embedding_dim {
					let weight_index = hidden_index * self.embedding.embedding_dim + embedding_index;

                    self.rnn_cell.grad_w_ih[weight_index] +=
                        hidden_gradient[hidden_index] * embedding[embedding_index];
                }

                // Hidden weights gradient
                for previous_index in 0..self.rnn_cell.hidden_size {
                	let weight_index = hidden_index * self.rnn_cell.hidden_size + previous_index;

                    self.rnn_cell.grad_w_hh[weight_index] += hidden_gradient[hidden_index] * previous_hidden[previous_index];
                }
            }

            // Compute gradient for embedded inputs at this time step
            for embedding_index in 0..self.embedding.embedding_dim {
                for hidden_index in 0..self.rnn_cell.hidden_size {
                	let weight_index = hidden_index * self.embedding.embedding_dim + embedding_index;

                    embedded_gradients[time][embedding_index] +=
                        hidden_gradient[hidden_index] * self.rnn_cell.w_ih[weight_index];
                }
            }

            // Compute gradient for previous hidden state
            let mut previous_gradient = vec![0.0; self.rnn_cell.hidden_size];

            for previous_index in 0..self.rnn_cell.hidden_size {
                for hidden_index in 0..self.rnn_cell.hidden_size {
                	let weight_index = hidden_index * self.rnn_cell.hidden_size + previous_index;

                    previous_gradient[previous_index] += hidden_gradient[hidden_index] * self.rnn_cell.w_hh[weight_index];
                }
            }

            // Prepare for next iteration
            dh_next = previous_gradient;
        }

        // Backpropagate through embedding
        for (time, token) in self
            .cached_tokens
            .iter()
            .copied()
            .enumerate()
        {
        	let embedding_start = token * self.embedding.embedding_dim;

         	for embedding_index in 0..self.embedding.embedding_dim {
          		self.embedding.grad_weights[embedding_start + embedding_index] += embedded_gradients[time][embedding_index];
          	}
        }

        self.cached_tokens.clone()
    }

    fn gradients(&self) -> Vec<f32> {
        let mut gradients = Vec::new();

        // Collect embedding gradients
        gradients.extend(&self.embedding.grad_weights);

        // Collect RNN gradients
        gradients.extend(&self.rnn_cell.grad_w_ih);
        gradients.extend(&self.rnn_cell.grad_w_hh);
        gradients.extend(&self.rnn_cell.grad_b_h);

        gradients
    }

    fn apply_gradients(&mut self, gradients: &[f32], learning_rate: f32) {
		let expected = self.gradients().len();

		assert_eq!(
			gradients.len(),
			expected,
			"Text gradient size mismatch: expected {}, got {}",
			expected,
			gradients.len()
		);

    	let mut offset = 0usize;

  		for weight in &mut self.embedding.weights {
            *weight -=
                learning_rate * gradients[offset];

            offset += 1;
        }

        for weight in &mut self.rnn_cell.w_ih {
            *weight -=
                learning_rate * gradients[offset];

            offset += 1;
        }

        for weight in &mut self.rnn_cell.w_hh {
            *weight -=
                learning_rate * gradients[offset];

            offset += 1;
        }

        for bias in &mut self.rnn_cell.b_h {
            *bias -=
                learning_rate * gradients[offset];

            offset += 1;
        }
    }

    fn zero_grad(&mut self) {
        self.embedding
            .grad_weights
           	.fill(0.0);

        self.rnn_cell
            .grad_w_ih
           	.fill(0.0);

        self.rnn_cell
            .grad_w_hh
           	.fill(0.0);

        self.rnn_cell
            .grad_b_h
            .fill(0.0);
    }
}
