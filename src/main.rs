mod layer;
mod network;

use ndarray::{Array1};
use plotters::prelude::*;

fn searched_function(x: f64) -> f64 {
    x.powi(2)
}

fn plot(xs: &[f64], y_nn: &[f64], y_true: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

    let root = BitMapBackend::new("curve.png", (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("f:x vs Neural Network", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;

    chart.configure_mesh().draw()?;

    // -------------------------
    // VRAIE FONCTION (ligne bleue)
    // -------------------------
    chart.draw_series(LineSeries::new(
        xs.iter().zip(y_true.iter()).map(|(&x, &y)| (x, y)),
        &BLUE,
    ))?
        .label("x²")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 10, y)], &BLUE));

    // -------------------------
    // RÉSEAU (points rouges)
    // -------------------------
    chart.draw_series(
        xs.iter().zip(y_nn.iter()).map(|(&x, &y)| {
            Circle::new((x, y), 3, RED.filled())
        }),
    )?
        .label("NN")
        .legend(|(x, y)| Circle::new((x, y), 3, RED.filled()));

    // -------------------------
    // LÉGENDE
    // -------------------------
    chart.configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn main() {
    let conns = [1, 10, 20, 10, 40, 1];
    let mut network = network::Network::new(&conns);

    const TRAINING_SIZE: usize = 1000;
    const EPOCHS: usize = 1000;

    let training_values: Vec<f64> =
        (0..TRAINING_SIZE).map(|_| rand::random::<f64>()).collect();

    let right_training_values_outputs: Vec<f64> =
        training_values.iter().map(|&x| searched_function(x)).collect();

    for epoch in 0..EPOCHS {
        let mut loss = 0.0;

        for i in 0..TRAINING_SIZE {
            let input = Array1::from_vec(vec![training_values[i]]);
            let output = network.forward(input);

            let err = output[0] - right_training_values_outputs[i];
            loss += err * err;

            network.backward(Array1::from_vec(vec![2.0 * err]));
        }

        loss /= TRAINING_SIZE as f64;

        if epoch % 100 == 0 {
            println!("Epoch : {} Loss: {}", epoch, loss);
        }
    }

    // -------------------------
    // TEST + DATA FOR PLOT
    // -------------------------

    let mut xs = Vec::new();
    let mut ys_nn = Vec::new();
    let mut ys_true = Vec::new();

    for i in 0..100 {
        let x = i as f64 / 100.0;

        let input = Array1::from_vec(vec![x]);
        let output = network.forward(input);

        xs.push(x);
        ys_nn.push(output[0]);
        ys_true.push(searched_function(x));
    }

    plot(&xs, &ys_nn, &ys_true).unwrap();
}