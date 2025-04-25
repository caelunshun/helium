use helium::{Device, Param, Tensor};
use rand::prelude::*;
use rand_distr::Normal;
use rand_pcg::Pcg64Mcg;

/// Gradient descent on a linear regression task.
fn main() {
    let mut rng = Pcg64Mcg::seed_from_u64(6666);
    let device = Device::Cuda(0);

    struct DataPoint {
        inputs: [f32; 8],
        output: f32,
    }

    let slopes = [(); 8].map(|_| rng.random_range(0.1f32..=20.0));

    let input_distr = Normal::new(50.0, 25.0).unwrap();
    let noise_distr = Normal::new(0.0, 2.0).unwrap();
    let data_points = (0..100_000)
        .map(|_| {
            let inputs: [f32; 8] = [(); 8].map(|_| input_distr.sample(&mut rng));
            let output = inputs
                .iter()
                .copied()
                .zip(slopes)
                .map(|(input, slope)| input * slope)
                .sum::<f32>();
            let output = output + noise_distr.sample(&mut rng);
            DataPoint { inputs, output }
        })
        .collect::<Vec<_>>();

    let weights_distr = Normal::new(0.0, 1.0).unwrap();
    let weights: Vec<f32> = (0..8).map(|_| weights_distr.sample(&mut rng)).collect();
    let mut weights = Param::new(Tensor::<2>::from_slice(weights, [1, 8], device));

    for _epoch in 0..50 {
        let batch_size = 1024;
        for batch in data_points.chunks_exact(batch_size) {
            let input: Vec<f32> = batch.iter().flat_map(|data| data.inputs).collect();
            let target: Vec<f32> = batch.iter().map(|data| data.output).collect();

            let input = Tensor::<2>::from_slice(input, [batch_size, 8], device);
            let target = Tensor::<2>::from_slice(target, [batch_size, 1], device);

            let result = weights.value().matmul(input.transpose()).transpose();
            let loss = (result - target).pow_scalar(2.0).reduce_mean::<1>(2);

            let grads = loss.clone().backward();
            let grad = grads.get::<2>(weights.id());
            weights.set_value(weights.value().clone() - grad * 1e-5);

            println!("Loss for this batch: {:.2}", loss.to_scalar::<f32>());
        }
    }

    let weights = weights.into_value().to_vec::<f32>();
    let weight_loss = weights
        .iter()
        .copied()
        .zip(slopes)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / 8.0;
    println!("Mean absolute difference between weights and targets: {weight_loss:.4}");
}
