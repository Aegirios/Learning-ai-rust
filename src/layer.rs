use ndarray::{Array, Array1, Array2};

pub struct Layer {
    pub weight_matrix: Array2<f64>,
    pub bias_vector: Array1<f64>,
}

impl Layer {
    pub fn new(n_inputs: usize, n_outputs: usize) -> Self {
        Self {
            weight_matrix: Array2::from_shape_fn((n_outputs, n_inputs), |_| rand::random::<f64>() * 0.1),
            bias_vector: Array1::zeros(n_outputs),
        }
    }

    pub fn forward(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        (&inputs.dot(&self.weight_matrix) + &self.bias_vector).map(|x| x.max(0.0))
    }
}