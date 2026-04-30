mod layer;
mod network;

use ndarray::Array;
use rand::prelude::*;

fn main() {
    let conns = [1, 10, 20, 5, 1];
    let mut network = network::Network::new(&conns);

    let input = (0..conns[0]).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
    let output = network.forward(Array::from_vec(input));

    println!("Output: {:?}", output);
}