use rand::prelude::*;

fn main() {
    let mut rng = rand::rng();
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for _ in 0..100 {
        let x: f64 = rng.random_range(0.0..10.0);
        let noise: f64 = rng.random_range(-1.0..1.0);
        let y = 2.0 * x + 1.0 + noise;

        x_data.push(x);
        y_data.push(y);
    }

    let mut w: f64 = 0.0;
    let mut b: f64 = 0.0;

    let learning_rate = 0.001;
    let epochs = 2000;

    for epoch in 0..epochs {
        let mut dw = 0.0;
        let mut db = 0.0;
        let n = x_data.len() as f64;

        for i in 0..x_data.len() {
            let x = x_data[i];
            let y = y_data[i];

            let y_pred = w * x + b;
            let error = y_pred - y;

            dw += error * x;
            db += error;
        }

        dw /= n;
        db /= n;

        w -= learning_rate * dw;
        b -= learning_rate * db;

        if epoch % 100 == 0 {
            println!("Epoch {}: w = {}, b = {}", epoch, w, b);
        }
    }

    println!("Modèle final: y = {}x + {}", w, b);
}