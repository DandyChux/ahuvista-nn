use crate::core::{Layer, Module};
use ndarray::{s, Array, Array1, Array3, Array4, Axis};
use rand::Rng;

/// 2D convolution layer.
///
/// Applies convolution operation over a 3D input (channels × height × width).
#[derive(Debug)]
pub struct Conv2DLayer {
    /// Number of input channels
    in_channels: usize,

    /// Number of output channels
    out_channels: usize,

    /// Kernel size (square kernels)
    kernel_size: usize,

    /// Convolution stride
    stride: usize,

    /// Padding size
    padding: usize,

    /// Weights: [out_channels, in_channels, kernel_size, kernel_size]
    weights: Array4<f32>,

    /// Biases: [out_channels]
    biases: Array1<f32>,

    /// Gradients for weights and biases
    grad_weights: Array4<f32>,
    grad_biases: Array1<f32>,

    /// Cache for backpropagation
    last_input: Option<Array3<f32>>,
    last_output: Option<Array3<f32>>,
}

impl Conv2DLayer {
    /// Create a new 2D convolution layer.
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolutional kernel (square)
    /// * `stride` - Convolution stride
    /// * `padding` - Padding size
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();

        // He initialization for ReLU activation
        let std_dev = (2.0 / (in_channels * kernel_size * kernel_size) as f32).sqrt();

        let weights_shape = [out_channels, in_channels, kernel_size, kernel_size];
        let weights = Array::from_shape_fn(weights_shape, |_| rng.random_range(-std_dev..std_dev));

        let biases = Array1::zeros(out_channels);

        Conv2DLayer {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            biases,
            grad_weights: Array::zeros(weights_shape),
            grad_biases: Array1::zeros(out_channels),
            last_input: None,
            last_output: None,
        }
    }

    /// Get all parameters from the layer.
    pub fn get_parameters(&self) -> Vec<f32> {
        let mut params = self.weights.iter().copied().collect::<Vec<f32>>();
        params.extend(self.biases.iter().copied());
        params
    }

    /// Set all parameters in the layer.
    /// Returns the number of parameters consumed.
    pub fn set_parameters(&mut self, params: &[f32]) -> Result<usize, Box<dyn std::error::Error>> {
        let weights_size = self.weights.len();
        let biases_size = self.biases.len();
        let expected_len = weights_size + biases_size;

        if params.len() < expected_len {
            return Err(format!(
                "Not enough parameters. Expected {}, got {}",
                expected_len,
                params.len()
            )
            .into());
        }

        self.weights = Array::from_shape_vec(self.weights.dim(), params[..weights_size].to_vec())?;
        self.biases = Array1::from_vec(params[weights_size..expected_len].to_vec());

        Ok(expected_len)
    }

    /// Calculate output dimensions after convolution.
    ///
    /// # Arguments
    /// * `input_height` - Height of input
    /// * `input_width` - Width of input
    ///
    /// # Returns
    /// (output_height, output_width)
    fn output_dimensions(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let output_height = (input_height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_width = (input_width + 2 * self.padding - self.kernel_size) / self.stride + 1;
        (output_height, output_width)
    }

    /// ReLU activation function.
    fn relu(x: &Array3<f32>) -> Array3<f32> {
        x.mapv(|v| v.max(0.0))
    }

    /// Derivative of ReLU.
    fn relu_derivative(x: &Array3<f32>) -> Array3<f32> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    /// Apply padding to input.
    fn pad_input(&self, input: &Array3<f32>) -> Array3<f32> {
        if self.padding == 0 {
            return input.clone();
        }

        let (c, h, w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let new_h = h + 2 * self.padding;
        let new_w = w + 2 * self.padding;

        let mut padded = Array3::zeros((c, new_h, new_w));

        // Copy input to padded array (central part)
        for i in 0..c {
            for j in 0..h {
                for k in 0..w {
                    padded[[i, j + self.padding, k + self.padding]] = input[[i, j, k]];
                }
            }
        }

        padded
    }

    /// Forward pass for a single window.
    fn forward_window(&self, input_window: &Array3<f32>) -> Array1<f32> {
        let mut output = Array1::zeros(self.out_channels);

        for out_ch in 0..self.out_channels {
            let mut sum = 0.0;

            for in_ch in 0..self.in_channels {
                for i in 0..self.kernel_size {
                    for j in 0..self.kernel_size {
                        sum += input_window[[in_ch, i, j]] * self.weights[[out_ch, in_ch, i, j]];
                    }
                }
            }

            output[out_ch] = sum + self.biases[out_ch];
        }

        output
    }
}

impl Module<Array3<f32>, Array3<f32>> for Conv2DLayer {
    fn forward(&mut self, input: Array3<f32>) -> Array3<f32> {
        let (channels, height, width) = (input.shape()[0], input.shape()[1], input.shape()[2]);

        assert_eq!(channels, self.in_channels, "Input channels don't match");

        // Apply padding
        let padded_input = self.pad_input(&input);
        let padded_height = padded_input.shape()[1];
        let padded_width = padded_input.shape()[2];

        // Calculate output dimensions
        let (output_height, output_width) = self.output_dimensions(height, width);

        // Prepare output array
        let mut output = Array3::zeros((self.out_channels, output_height, output_width));

        // Sliding window convolution
        for i in 0..output_height {
            let start_i = i * self.stride;

            for j in 0..output_width {
                let start_j = j * self.stride;

                // Extract window
                let mut window =
                    Array3::zeros((self.in_channels, self.kernel_size, self.kernel_size));

                for c in 0..self.in_channels {
                    for ki in 0..self.kernel_size {
                        for kj in 0..self.kernel_size {
                            window[[c, ki, kj]] = padded_input[[c, start_i + ki, start_j + kj]];
                        }
                    }
                }

                // Forward pass for this window
                let window_output = self.forward_window(&window);

                // Store results
                for out_ch in 0..self.out_channels {
                    output[[out_ch, i, j]] = window_output[out_ch];
                }
            }
        }

        // Cache for backward pass
        self.last_input = Some(input);

        // Apply ReLU activation
        let activated = Self::relu(&output);
        self.last_output = Some(activated.clone());

        activated
    }

    fn backward(&mut self, grad_output: Array3<f32>) -> Array3<f32> {
        // Ensure we have cached values
        let input = self.last_input.as_ref().expect("No cached input");
        let output = self.last_output.as_ref().expect("No cached output");

        // Backpropagate through ReLU
        let relu_grad = Self::relu_derivative(output);
        let mut grad_pre_activation = &grad_output * &relu_grad;

        // Get dimensions
        let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);

        let (out_c, out_h, out_w) = (
            grad_pre_activation.shape()[0],
            grad_pre_activation.shape()[1],
            grad_pre_activation.shape()[2],
        );

        // Prepare gradients
        let mut grad_input = Array3::zeros((in_c, in_h, in_w));

        // Pad input for easier window extraction
        let padded_input = self.pad_input(input);

        // Compute gradients
        for out_ch in 0..out_c {
            // Accumulate bias gradients
            self.grad_biases[out_ch] += grad_pre_activation.slice(s![out_ch, .., ..]).sum();

            // Loop through each position in the output
            for i in 0..out_h {
                let start_i = i * self.stride;

                for j in 0..out_w {
                    let start_j = j * self.stride;
                    let grad_val = grad_pre_activation[[out_ch, i, j]];

                    // Update weight gradients
                    for in_ch in 0..in_c {
                        for ki in 0..self.kernel_size {
                            for kj in 0..self.kernel_size {
                                let input_val = padded_input[[in_ch, start_i + ki, start_j + kj]];
                                self.grad_weights[[out_ch, in_ch, ki, kj]] += grad_val * input_val;
                            }
                        }
                    }

                    // Update input gradients (if padding is 0 for simplicity)
                    if self.padding == 0 {
                        for in_ch in 0..in_c {
                            for ki in 0..self.kernel_size {
                                for kj in 0..self.kernel_size {
                                    if start_i + ki < in_h && start_j + kj < in_w {
                                        grad_input[[in_ch, start_i + ki, start_j + kj]] +=
                                            grad_val * self.weights[[out_ch, in_ch, ki, kj]];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        grad_input
    }

    fn gradients(&self) -> Vec<f32> {
        let mut grads = Vec::new();
        // Flatten weight gradients
        grads.extend(self.grad_weights.iter().cloned());
        // Add bias gradients
        grads.extend(self.grad_biases.iter().cloned());
        grads
    }

    fn apply_gradients(&mut self, gradients: &[f32], learning_rate: f32) {
        let weights_len = self.grad_weights.len();

        // Apply weight gradients
        for (i, grad) in gradients[..weights_len].iter().enumerate() {
            self.weights.as_slice_mut().unwrap()[i] -= grad * learning_rate;
        }

        // Apply bias gradients
        for (i, grad) in gradients[weights_len..].iter().enumerate() {
            self.biases[i] -= grad * learning_rate;
        }
    }

    fn zero_grad(&mut self) {
        self.grad_weights.fill(0.0);
        self.grad_biases.fill(0.0);
    }
}

/// Max pooling layer.
///
/// Reduces spatial dimensions by taking maximum value in each window.
#[derive(Debug)]
pub struct MaxPool2DLayer {
    /// Pool size (square)
    pool_size: usize,

    /// Stride
    stride: usize,

    /// Cache for backpropagation
    last_input: Option<Array3<f32>>,
    max_indices: Option<Array3<(usize, usize)>>,
}

impl MaxPool2DLayer {
    /// Create a new max pooling layer.
    ///
    /// # Arguments
    /// * `pool_size` - Size of pooling window
    /// * `stride` - Stride (if None, use pool_size)
    pub fn new(pool_size: usize, stride: Option<usize>) -> Self {
        MaxPool2DLayer {
            pool_size,
            stride: stride.unwrap_or(pool_size),
            last_input: None,
            max_indices: None,
        }
    }

    /// Calculate output dimensions after pooling.
    ///
    /// # Arguments
    /// * `input_height` - Height of input
    /// * `input_width` - Width of input
    ///
    /// # Returns
    /// (output_height, output_width)
    fn output_dimensions(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let output_height = (input_height - self.pool_size) / self.stride + 1;
        let output_width = (input_width - self.pool_size) / self.stride + 1;
        (output_height, output_width)
    }
}

impl Module<Array3<f32>, Array3<f32>> for MaxPool2DLayer {
    fn forward(&mut self, input: Array3<f32>) -> Array3<f32> {
        let (channels, height, width) = (input.shape()[0], input.shape()[1], input.shape()[2]);

        // Calculate output dimensions
        let (output_height, output_width) = self.output_dimensions(height, width);

        // Prepare output arrays
        let mut output = Array3::zeros((channels, output_height, output_width));
        let mut max_indices = Array3::from_elem((channels, output_height, output_width), (0, 0));

        // Apply max pooling
        for c in 0..channels {
            for i in 0..output_height {
                let start_i = i * self.stride;

                for j in 0..output_width {
                    let start_j = j * self.stride;

                    // Find maximum in pool window
                    let mut max_val = f32::MIN;
                    let mut max_idx = (0, 0);

                    for pi in 0..self.pool_size {
                        for pj in 0..self.pool_size {
                            let input_val = input[[c, start_i + pi, start_j + pj]];
                            if input_val > max_val {
                                max_val = input_val;
                                max_idx = (pi, pj);
                            }
                        }
                    }

                    output[[c, i, j]] = max_val;
                    max_indices[[c, i, j]] = max_idx;
                }
            }
        }

        // Cache for backward pass
        self.last_input = Some(input);
        self.max_indices = Some(max_indices);

        output
    }

    fn backward(&mut self, grad_output: Array3<f32>) -> Array3<f32> {
        // Ensure we have cached values
        let input = self.last_input.as_ref().expect("No cached input");
        let indices = self.max_indices.as_ref().expect("No cached indices");

        let (channels, height, width) = (input.shape()[0], input.shape()[1], input.shape()[2]);

        let (out_c, out_h, out_w) = (
            grad_output.shape()[0],
            grad_output.shape()[1],
            grad_output.shape()[2],
        );

        // Prepare gradient for input
        let mut grad_input = Array3::zeros((channels, height, width));

        // Backpropagate through max pooling
        for c in 0..out_c {
            for i in 0..out_h {
                let start_i = i * self.stride;

                for j in 0..out_w {
                    let start_j = j * self.stride;

                    // Get max index for this output
                    let (max_i, max_j) = indices[[c, i, j]];

                    // Only the maximum receives gradient
                    grad_input[[c, start_i + max_i, start_j + max_j]] += grad_output[[c, i, j]];
                }
            }
        }

        grad_input
    }

    fn gradients(&self) -> Vec<f32> {
        // No learnable parameters
        Vec::new()
    }

    fn apply_gradients(&mut self, _gradients: &[f32], _learning_rate: f32) {
        // No parameters to update
    }

    fn zero_grad(&mut self) {
        // No gradients to zero
    }
}

/// Network for processing image data.
///
/// Uses a small convolutional neural network optimized for
/// low-compute environments to extract features from images.
#[derive(Debug)]
pub struct ImageNet {
    /// First convolution layer
    conv1: Conv2DLayer,

    /// First pooling layer
    pool1: MaxPool2DLayer,

    /// Second convolution layer
    conv2: Conv2DLayer,

    /// Second pooling layer
    pool2: MaxPool2DLayer,

    /// Flattened input layer
    fc_layer: Layer,

    /// Input dimensions (channels, height, width)
    input_dims: (usize, usize, usize),

    /// Output feature dimensions after flattening
    output_size: usize,

    /// Last flattened output for backpropagation
    last_flattened: Vec<f32>,
}

impl ImageNet {
    /// Create a new image network.
    ///
    /// # Arguments
    /// * `input_channels` - Number of input channels (e.g. 3 for RGB)
    /// * `input_height` - Height of input images
    /// * `input_width` - Width of input images
    /// * `feature_size` - Size of output feature vector
    pub fn new(input_channels: usize, input_height: usize, input_width: usize) -> Self {
        // Define a small, efficient CNN
        let conv1 = Conv2DLayer::new(input_channels, 8, 3, 1, 1);
        let pool1 = MaxPool2DLayer::new(2, None);

        let conv2 = Conv2DLayer::new(8, 16, 3, 1, 1);
        let pool2 = MaxPool2DLayer::new(2, None);

        // Calculate output dimensions
        let (h1, w1) = conv1.output_dimensions(input_height, input_width);
        let (h2, w2) = pool1.output_dimensions(h1, w1);
        let (h3, w3) = conv2.output_dimensions(h2, w2);
        let (h4, w4) = pool2.output_dimensions(h3, w3);

        let flattened_size = 32 * h4 * w4;
        let output_size = 128;

        // Final output size after flattening
        let output_size = 16 * h4 * w4;

        ImageNet {
            conv1,
            pool1,
            conv2,
            pool2,
            fc_layer: Layer::new(output_size, flattened_size),
            input_dims: (input_channels, input_height, input_width),
            output_size,
            last_flattened: Vec::new(),
        }
    }

    /// Get all parameters from the network.
    pub fn get_parameters(&self) -> Vec<f32> {
        let mut params = self.conv1.get_parameters();
        params.extend(self.conv2.get_parameters());
        params.extend(self.fc_layer.get_parameters());
        params
    }

    /// Set all parameters in the network.
    /// Returns the number of parameters consumed.
    pub fn set_parameters(&mut self, params: &[f32]) -> Result<usize, Box<dyn std::error::Error>> {
        let mut offset = 0;
        offset += self.conv1.set_parameters(&params[offset..])?;
        offset += self.conv2.set_parameters(&params[offset..])?;
        offset += self.fc_layer.set_parameters(&params[offset..])?;
        Ok(offset)
    }

    /// Get the output size of this network
    pub fn output_size(&self) -> usize {
        self.output_size
    }
}

impl Module<Array3<f32>, Vec<f32>> for ImageNet {
    fn forward(&mut self, input: Array3<f32>) -> Vec<f32> {
        // Check input dimensions
        assert_eq!(
            input.shape()[0],
            self.input_dims.0,
            "Input channels don't match"
        );
        assert_eq!(
            input.shape()[1],
            self.input_dims.1,
            "Input height doesn't match"
        );
        assert_eq!(
            input.shape()[2],
            self.input_dims.2,
            "Input width doesn't match"
        );

        // Forward through convolutional layers
        let x = self.conv1.forward(input);
        let x = self.pool1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.pool2.forward(x);

        // Flatten to 1D vector
        let flattened: Vec<f32> = x.iter().cloned().collect();
        self.last_flattened = flattened.clone();

        flattened
    }

    fn backward(&mut self, grad_output: Vec<f32>) -> Array3<f32> {
        // Reshape gradient back to 3D
        let (c, h, w) = {
            let (h1, w1) = self
                .conv1
                .output_dimensions(self.input_dims.1, self.input_dims.2);
            let (h2, w2) = self.pool1.output_dimensions(h1, w1);
            let (h3, w3) = self.conv2.output_dimensions(h2, w2);
            let (h4, w4) = self.pool2.output_dimensions(h3, w3);
            (16, h4, w4)
        };

        let mut grad_3d = Array3::zeros((c, h, w));
        let flat_grad = grad_output.as_slice();

        let mut idx = 0;
        for i in 0..c {
            for j in 0..h {
                for k in 0..w {
                    grad_3d[[i, j, k]] = flat_grad[idx];
                    idx += 1;
                }
            }
        }

        // Backpropagate through layers
        let grad_pool2 = self.pool2.backward(grad_3d);
        let grad_conv2 = self.conv2.backward(grad_pool2);
        let grad_pool1 = self.pool1.backward(grad_conv2);
        let grad_input = self.conv1.backward(grad_pool1);

        grad_input
    }

    fn gradients(&self) -> Vec<f32> {
        let mut grads = Vec::new();
        grads.extend(self.conv1.gradients());
        grads.extend(self.pool1.gradients());
        grads.extend(self.conv2.gradients());
        grads.extend(self.pool2.gradients());
        grads
    }

    fn apply_gradients(&mut self, gradients: &[f32], learning_rate: f32) {
        let conv1_size = self.conv1.gradients().len();
        let pool1_size = self.pool1.gradients().len();
        let conv2_size = self.conv2.gradients().len();

        let mut offset = 0;

        self.conv1
            .apply_gradients(&gradients[offset..offset + conv1_size], learning_rate);
        offset += conv1_size;

        self.pool1
            .apply_gradients(&gradients[offset..offset + pool1_size], learning_rate);
        offset += pool1_size;

        self.conv2
            .apply_gradients(&gradients[offset..offset + conv2_size], learning_rate);
        offset += conv2_size;

        self.pool2
            .apply_gradients(&gradients[offset..], learning_rate);
    }

    fn zero_grad(&mut self) {
        self.conv1.zero_grad();
        self.conv2.zero_grad();
        // Pooling layers have no gradients to zero
    }
}
