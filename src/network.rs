use ndarray::{Array1};
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

    pub fn backward(&mut self, grad_output: Array1<f64>) {
        let mut grad = grad_output;

        for layer in self.layers.iter_mut().rev() {
            let (dw, db, grad_input) = layer.backward(&grad);

            layer.apply_gradients(&dw, &db, 0.01);

            grad = grad_input;
        }
    }
}