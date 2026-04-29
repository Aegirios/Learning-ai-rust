use ndarray::Array;
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
}