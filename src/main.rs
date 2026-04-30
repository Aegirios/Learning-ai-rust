use std::io;

mod layer;
mod network;

use ndarray::{Array, Array1};
fn main() {
    let conns = [1, 10, 20, 10, 1];
    let mut network = network::Network::new(&conns);

    const TRAINING_SIZE:usize = 1000;
    const EPOCHS:usize = 1000;
    fn searched_function(x: f64) -> f64 {
        x.powi(2)
    }

    let training_values = (0..TRAINING_SIZE).map(|_| rand::random::<f64>()).collect::<Vec<_>>();
    let right_training_values_outputs = (0..TRAINING_SIZE).map(|i| searched_function(training_values[i])).collect::<Vec<_>>();

    for epoch in 0..EPOCHS {
        let mut loss = 0.0;
        for i in 0..TRAINING_SIZE {
            let input = Array::from_vec(vec![training_values[i]]);
            let output = network.forward(input);
            loss += (right_training_values_outputs[i] - output[0]).powi(2);
            network.backward(Array::from_vec(vec![2.0 * (output[0] - right_training_values_outputs[i])]));
        }

        loss /= TRAINING_SIZE as f64;

        if epoch % 100 == 0 {
            println!("Epoch : {epoch} Loss: {loss}");
        }
    }

    let mut input = String::new();

    println!("valeur à tester :");

    io::stdin().read_line(&mut input).unwrap();

    let x: f64 = match input.trim().parse() {
        Ok(v) => v,
        Err(_) => {
            println!("un nombre bordel");
            return;
        }
    };

    let input_array = Array1::from_vec(vec![x]);

    let output = network.forward(input_array);

    println!("Résultat du réseau : {}", output[0]);
}