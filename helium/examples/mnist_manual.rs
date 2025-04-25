use helium::{
    Device, Tensor,
    initializer::Initializer,
    modules::linear::{Linear, LinearSettings},
    optimizer::{Adam, AdamSettings, Optimizer},
};
use helium_macros::Module;
use mnist::MnistBuilder;
use pollster::FutureExt;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::{future::Future, pin::Pin, thread, time::Instant};

#[derive(Module)]
struct Model {
    layers: [Linear; 3],
}

impl Model {
    pub fn with_random_weights(rng: &mut impl Rng, device: Device) -> Self {
        Self {
            layers: [
                Linear::new(
                    LinearSettings {
                        in_features: 28 * 28,
                        out_features: 256,
                        weight_initializer: Initializer::KaimingNormal,
                        ..Default::default()
                    },
                    rng,
                    device,
                ),
                Linear::new(
                    LinearSettings {
                        in_features: 256,
                        out_features: 128,
                        weight_initializer: Initializer::KaimingNormal,
                        ..Default::default()
                    },
                    rng,
                    device,
                ),
                Linear::new(
                    LinearSettings {
                        in_features: 128,
                        out_features: 10,
                        weight_initializer: Initializer::KaimingNormal,
                        ..Default::default()
                    },
                    rng,
                    device,
                ),
            ],
        }
    }

    pub fn forward(&self, mut x: Tensor<2>) -> Tensor<2> {
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            if i != self.layers.len() - 1 {
                x = x.relu();
            }
        }
        x
    }
}

fn log_softmax(x: Tensor<2>) -> Tensor<2> {
    let max = x.reduce_max::<2>(1).broadcast_to(x.shape());
    &x - &max
        - (&x - max)
            .exp()
            .reduce_sum::<2>(1)
            .log()
            .broadcast_to(x.shape())
}

fn cross_entropy_loss(logits: Tensor<2>, targets: Tensor<2>) -> Tensor<1> {
    -(log_softmax(logits) * targets).reduce_mean::<1>(2)
}

#[derive(Debug, Clone)]
struct Item {
    image: Vec<f32>,
    label_one_hot: Vec<f32>,
}

fn main() {
    tracing_subscriber::fmt::init();
    let device = Device::Cuda(0);
    let mut rng = Pcg64Mcg::seed_from_u64(666);
    let mut model = Model::with_random_weights(&mut rng, device);

    let mnist = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(60_000)
        .validation_set_length(10_000)
        .test_set_length(0)
        .download_and_extract()
        .finalize();

    let mut items: Vec<Item> = mnist
        .trn_img
        .chunks_exact(28 * 28)
        .zip(mnist.trn_lbl.chunks_exact(10))
        .map(|(img, label)| Item {
            image: img
                .iter()
                .copied()
                .map(|x| x as f32 / u8::MAX as f32)
                .collect(),
            label_one_hot: label.iter().copied().map(|x| x as f32).collect(),
        })
        .collect();
    let validation_items: Vec<Item> = mnist
        .val_img
        .chunks_exact(28 * 28)
        .zip(mnist.val_lbl.chunks_exact(10))
        .map(|(img, label)| Item {
            image: img
                .iter()
                .copied()
                .map(|x| x as f32 / u8::MAX as f32)
                .collect(),
            label_one_hot: label.iter().copied().map(|x| x as f32).collect(),
        })
        .collect();

    let num_epochs = 20;
    let mut lr = 1e-3;
    let lr_gamma = 0.97;
    let batch_size = 1024;

    let (loss_tx, loss_rx) = flume::bounded::<Pin<Box<dyn Future<Output = f32> + Send>>>(2);
    let blocker_thread = thread::spawn(move || {
        for loss in loss_rx {
            let loss = loss.block_on();
            println!("Training batch loss: {loss:.3}");
        }
    });

    let mut optimizer = Adam::new(AdamSettings::default());

    for _epoch in 0..num_epochs {
        items.shuffle(&mut rng);

        let mut prev = Instant::now();
        for batch in items.chunks_exact(batch_size) {
            let input: Vec<f32> = batch
                .iter()
                .flat_map(|item| item.image.as_slice())
                .copied()
                .collect();
            let labels: Vec<f32> = batch
                .iter()
                .flat_map(|item| item.label_one_hot.as_slice())
                .copied()
                .collect();

            let start = Instant::now();

            let input = Tensor::<2>::from_slice(input, [batch_size, 28 * 28], device);
            let labels = Tensor::<2>::from_slice(labels, [batch_size, 10], device);

            let logits = model.forward(input);
            let loss = cross_entropy_loss(logits, labels);

            let grads = loss.backward();
            optimizer.step(&mut model, &grads, lr);

            loss.async_start_eval();
            println!("Recorded in {:.2?}", start.elapsed());

            loss_tx
                .send(Box::pin(async move { loss.to_scalar_async::<f32>().await }))
                .unwrap();
            println!("Latency: {:.2?}", prev.elapsed());
            prev = Instant::now();
        }
        lr *= lr_gamma;
    }

    drop(loss_tx);
    blocker_thread.join().unwrap();

    // Validation
    let inputs: Vec<f32> = validation_items
        .iter()
        .flat_map(|item| item.image.as_slice())
        .copied()
        .collect();
    let inputs = Tensor::from_slice(inputs, [validation_items.len(), 28 * 28], device);

    let outputs = log_softmax(model.forward(inputs)).exp().to_vec::<f32>();

    let mut num_correct = 0;
    for (item, output) in validation_items.iter().zip(outputs.chunks_exact(10)) {
        let output = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        let expected = item.label_one_hot.iter().position(|&x| x == 1.0).unwrap();
        if output == expected {
            num_correct += 1;
        }
    }

    let accuracy = num_correct as f64 / validation_items.len() as f64 * 100.0;
    println!("Validation accuracy: {accuracy:.2}%");
}
