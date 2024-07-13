mod neuron;
mod layer;
mod network;
mod utils;

use layer::Layer;
use network::Network;

fn main() {
    // Define the structure of the network
    let mut network = Network::new();
    network.add_layer(Layer::new(3, 2)); // Hidden layer with 3 neurons, each with 2 inputs
    network.add_layer(Layer::new(1, 3)); // Output layer with 1 neuron, with 3 inputs from the hidden layer

    // Example input
    let inputs = vec![0.5, -1.0];

    // Forward pass
    let outputs = network.forward(&inputs);
    println!("Network output: {:?}", outputs);
}