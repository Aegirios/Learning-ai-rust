use ndarray::{Array, Array1, Ix1};
use crate::layer::Layer;

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(conns: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..conns.len() - 1 {
            layers.push(Layer::new(conns[i], conns[i + 1]));
        }
        Self { layers }
    }

    pub fn forward(&mut self, input: Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }
}