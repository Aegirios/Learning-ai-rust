use ndarray::{Array1, Array2};

pub struct Layer {
    pub weight_matrix: Array2<f64>,
    pub bias_vector: Array1<f64>,
    pub input_cache: Option<Array1<f64>>,
    pub z_cache: Option<Array1<f64>>,
}

impl Layer {
    pub fn new(n_inputs: usize, n_outputs: usize) -> Self {
        Self {
            weight_matrix: Array2::from_shape_fn((n_outputs, n_inputs), |_| rand::random::<f64>() * 0.1),
            bias_vector: Array1::zeros(n_outputs),
            input_cache: None,
            z_cache: None,
        }
    }

    pub fn forward(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        self.input_cache = Some(inputs.clone());
        let z_cache = &self.weight_matrix.dot(inputs) + &self.bias_vector;
        self.z_cache = Some(z_cache.clone());
        let output = z_cache.mapv(|x| x.max(0.0));
        output
    }

    pub fn backward(&self, grad_output: &Array1<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        // On a y_pred = ReLU(z) ; donc dL/dz = dL/dy_pred * dy_pred/dz
        // Or, dL/dy_pred = grad_output, et dy_pred/dz = 1 si z > 0, sinon 0
        // C'est à dire dL/dz = grad_output * ReLU'(z)
        // On note ReLU' grad_z
        let z = self.z_cache.as_ref().unwrap();
        let relu_mask = z.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        let grad_z = grad_output * &relu_mask;

        let input = self.input_cache.as_ref().unwrap();
        let dw = grad_z.view().insert_axis(ndarray::Axis(1))
            .dot(&input.view().insert_axis(ndarray::Axis(0)));

        let db = grad_z.clone();

        let grad_input = self.weight_matrix.t().dot(&grad_z);

        (dw, db, grad_input)
    }

    pub fn apply_gradients(&mut self, dw: &Array2<f64>, db: &Array1<f64>, lr: f64) {
        self.weight_matrix -= &(dw * lr);
        self.bias_vector -= &(db * lr);
    }


}