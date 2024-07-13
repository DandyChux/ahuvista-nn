use rand::Rng;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn initialize_weights(num_weights: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..num_weights).map(|_| rng.gen_range(-1.0..1.0)).collect()
}