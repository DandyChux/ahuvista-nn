#[cfg(test)]
mod tests {
    use ahuvista_nn::core::{Layer, Module, Network, Neuron};

    #[test]
    fn test_neuron_forward() {
        let mut neuron = Neuron::new(3);
        let inputs = vec![0.5, -1.0, 2.0];
        let output = neuron.forward(inputs);
        assert!(output >= 0.0 && output <= 1.0);
    }

    #[test]
    fn test_layer_forward() {
        let mut layer = Layer::new(3, 3);
        let inputs = vec![0.5, -1.0, 2.0];
        let outputs = layer.forward(inputs);
        assert_eq!(outputs.len(), 3);
        for output in outputs {
            assert!(output >= 0.0 && output <= 1.0);
        }
    }

    #[test]
    fn test_network_forward() {
        let mut network = Network::new();
        network.add_layer(Layer::new(3, 3));
        network.add_layer(Layer::new(2, 3));
        let inputs = vec![0.5, -1.0, 2.0];
        let outputs = network.forward(inputs);
        assert_eq!(outputs.len(), 2);
    }
}
