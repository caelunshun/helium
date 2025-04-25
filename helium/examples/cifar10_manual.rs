//! Training on CIFAR-10 image classification task,
//! with convolutional layers.
//!
//! Also demonstrates mixed-precision training. Model weights are
//! stored in `f32`, but activations and gradients are computed in `bf16`
//! which gives a ~2x speedup on Ampere and later GPUs.
//!
//! To download data:
//! ```bash
//! mkdir -p data && cd data
//! curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
//! tar -xvf cifar-10-binary.tar.gz
//! ```

use half::bf16;
use helium::{
    DataType, Device, Module, Tensor,
    conv::Conv2dParams,
    initializer::Initializer,
    loss::cross_entropy_loss,
    modules::{BatchNorm2d, Conv2d, batch_norm::ForwardMode, conv::Conv2dSettings},
    optimizer::{Optimizer, sgd::Sgd},
};
use rand::prelude::*;
use rand_distr::Normal;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use std::{
    fs::File,
    io,
    io::{BufReader, Read},
    path::Path,
    thread,
    time::Instant,
};

#[derive(Module)]
struct Model {
    layers: Vec<Conv2d>,
    batchnorms: Vec<BatchNorm2d>,
}

impl Model {
    pub fn new(rng: &mut impl Rng, device: Device) -> Self {
        fn make_settings(params: Conv2dParams) -> Conv2dSettings {
            Conv2dSettings {
                params,
                kernel_initializer: Initializer::KaimingNormal,
                bias: false,
                data_type: DataType::F32,
            }
        }

        let layers = vec![
            Conv2d::new(
                make_settings(Conv2dParams {
                    kernel_size: [3, 3],
                    in_channels: 3,
                    out_channels: 16,
                    ..Default::default()
                }),
                rng,
                device,
            ),
            Conv2d::new(
                make_settings(Conv2dParams {
                    kernel_size: [3, 3],
                    stride: [2, 2],
                    in_channels: 16,
                    out_channels: 32,
                    ..Default::default()
                }),
                rng,
                device,
            ),
            Conv2d::new(
                make_settings(Conv2dParams {
                    kernel_size: [3, 3],
                    stride: [2, 2],
                    in_channels: 32,
                    out_channels: 64,
                    ..Default::default()
                }),
                rng,
                device,
            ),
            Conv2d::new(
                make_settings(Conv2dParams {
                    kernel_size: [3, 3],
                    stride: [2, 2],
                    in_channels: 64,
                    out_channels: 64,
                    ..Default::default()
                }),
                rng,
                device,
            ),
            Conv2d::new(
                make_settings(Conv2dParams {
                    kernel_size: [3, 3],
                    stride: [1, 1],
                    in_channels: 64,
                    out_channels: 10,
                    ..Default::default()
                }),
                rng,
                device,
            ),
        ];
        let batchnorms = layers
            .iter()
            .take(layers.len() - 1)
            .map(|layer| BatchNorm2d::new(layer.params().out_channels, device))
            .collect();
        Self { layers, batchnorms }
    }

    pub fn forward(&mut self, mut x: Tensor<4>, mode: ForwardMode) -> Tensor<2> {
        let [batch_size, ..] = x.shape();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            if i != self.layers.len() - 1 {
                x = self.batchnorms[i].forward(&x, mode);
                x = x.relu();
            }
        }
        // Global average pooling
        x.swap_dims(2, 3)
            .swap_dims(1, 2)
            .reduce_mean::<3>(2)
            .reshape([batch_size, 10])
    }
}

const IMAGE_DIM: usize = 32;
const IMAGE_CHANNELS: usize = 3;

#[derive(Debug, Clone)]
struct DataItem {
    // HWC layout
    image: [bf16; IMAGE_DIM * IMAGE_DIM * IMAGE_CHANNELS],
    label: usize,
}

struct Batch {
    images: Tensor<4>,
    /// One-hot encoding
    labels: Tensor<2>,
}

impl Batch {
    pub fn new(items: &[DataItem], device: Device) -> Self {
        let images: Vec<bf16> = items.iter().flat_map(|item| item.image).collect();
        let labels_one_hot: Vec<bf16> = items
            .iter()
            .flat_map(|item| {
                let mut arr = [bf16::ZERO; 10];
                arr[item.label] = bf16::ONE;
                arr
            })
            .collect();
        Batch {
            images: Tensor::from_slice(
                images,
                [items.len(), IMAGE_DIM, IMAGE_DIM, IMAGE_CHANNELS],
                device,
            ),
            labels: Tensor::from_slice(labels_one_hot, [items.len(), 10], device),
        }
    }
}

fn load_data_file(path: &Path) -> Vec<DataItem> {
    // See https://www.cs.toronto.edu/~kriz/cifar.html
    // for binary file layout.
    let mut reader = BufReader::new(File::open(path).unwrap());

    let mut items = Vec::new();
    loop {
        let mut label_byte = [0u8; 1];
        match reader.read_exact(&mut label_byte) {
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => panic!("{e}"),
            _ => {}
        }

        let mut image = [0u8; IMAGE_DIM * IMAGE_DIM * IMAGE_CHANNELS];
        reader.read_exact(&mut image).unwrap();

        // Change from CHW layout to HWC
        let mut new_image = [0u8; IMAGE_DIM * IMAGE_DIM * IMAGE_CHANNELS];
        for y in 0..IMAGE_DIM {
            for x in 0..IMAGE_DIM {
                for channel in 0..IMAGE_CHANNELS {
                    new_image[channel + x * IMAGE_CHANNELS + y * IMAGE_CHANNELS * IMAGE_DIM] =
                        image[x + y * IMAGE_DIM + channel * IMAGE_DIM * IMAGE_DIM];
                }
            }
        }

        items.push(DataItem {
            image: new_image.map(|x| bf16::from_f32((x as f32 / 255.0).powf(2.2))),
            label: label_byte[0] as usize,
        });
    }
    items
}

fn augment_image(
    image: &mut [bf16],
    _height: usize,
    _width: usize,
    _num_channels: usize,
    rng: &mut impl Rng,
) {
    let noise_distr = Normal::new(0.0f32, rng.random_range(0.001..0.1)).unwrap();
    for sample in image {
        *sample = bf16::from_f32((sample.to_f32() + noise_distr.sample(rng)).clamp(0.0, 1.0));
    }
}

const TRAINING_DATA_FILES: &[&str] = &[
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
];
const VALIDATION_DATA_FILE: &str = "test_batch.bin";

fn main() {
    tracing_subscriber::fmt::init();

    if !Path::new("data/cifar-10-batches-bin").exists() {
        tracing::error!(
            "CIFAR-10 data not downloaded. Follow instructions at top of cifar10_manual.rs to download data."
        );
        return;
    }

    let start = Instant::now();
    let (mut training_data, validation_data) = rayon::join(
        || {
            TRAINING_DATA_FILES
                .par_iter()
                .flat_map(|name| {
                    let path = Path::new("data/cifar-10-batches-bin").join(name);
                    load_data_file(&path)
                })
                .collect::<Vec<_>>()
        },
        || {
            let path = Path::new("data/cifar-10-batches-bin").join(VALIDATION_DATA_FILE);
            load_data_file(&path)
        },
    );
    tracing::info!(
        "Loaded {} training items and {} validation items in {:.2?}",
        training_data.len(),
        validation_data.len(),
        start.elapsed()
    );

    let device = Device::Cuda(0);
    let mut rng = Pcg64Mcg::seed_from_u64(3377);
    let mut model = Model::new(&mut rng, device);

    // Initial learning rate at "warm-up" for first few epochs
    let mut lr = 1e-2;
    let lr_gamma = 0.995;
    let num_epochs = 200;
    let batch_size = 1024;

    // For profiling.
    let only_one_iter = false;

    training_data.shuffle(&mut rng);

    let mut optimizer = Sgd::new_with_momentum(0.9);

    thread::scope(|s| {
        for epoch in 0..num_epochs {
            tracing::info!("Starting epoch {epoch} with learning rate {lr}");

            if epoch == 5 {
                // End warmup phase
                lr = 1e0;
            }

            let (batch_tx, batch_rx) = flume::bounded(16);
            let training_data = &training_data;
            let mut reg_rng = Pcg64Mcg::from_rng(&mut rng);
            s.spawn(move || {
                let mut indexes = (0..training_data.len()).collect::<Vec<_>>();
                indexes.shuffle(&mut reg_rng);
                for index_chunk in indexes.chunks_exact(batch_size) {
                    let mut regularized_items: Vec<DataItem> = index_chunk
                        .iter()
                        .copied()
                        .map(|i| training_data[i].clone())
                        .collect();
                    regularized_items.iter_mut().for_each(|item| {
                        augment_image(
                            &mut item.image,
                            IMAGE_DIM,
                            IMAGE_DIM,
                            IMAGE_CHANNELS,
                            &mut reg_rng,
                        )
                    });

                    let batch = Batch::new(&regularized_items, device);
                    let labels = regularized_items
                        .iter()
                        .map(|item| item.label)
                        .collect::<Vec<_>>();
                    batch_tx.send((batch, labels)).ok();
                }
            });

            let (result_tx, result_rx) = flume::bounded::<(Vec<usize>, Tensor<2>, Tensor<1>)>(4);
            let thread = s.spawn(move || {
                for (labels, probs, loss) in result_rx {
                    let mut num_correct = 0;
                    for (label, probs) in labels
                        .iter()
                        .copied()
                        .zip(probs.to_vec::<f32>().chunks_exact(10))
                    {
                        let prediction = probs
                            .iter()
                            .copied()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.total_cmp(b))
                            .unwrap()
                            .0;
                        if prediction == label {
                            num_correct += 1;
                        }
                    }

                    tracing::info!(
                        "Epoch {epoch}, training batch loss {:.3}, accuracy {:.2}%",
                        loss.to_scalar::<f32>(),
                        num_correct as f64 / batch_size as f64 * 100.0
                    );
                }
            });

            let epoch_start = Instant::now();

            let mut num_batches = 0;
            for (batch, labels) in batch_rx {
                let start = Instant::now();
                let logits = model
                    .forward(batch.images, ForwardMode::Train)
                    .to_data_type(DataType::F32);
                let loss = cross_entropy_loss(logits.clone(), batch.labels);
                let probs = logits.log_softmax().exp();
                loss.async_start_eval();

                let grads = loss.backward();
                optimizer.step(&mut model, &grads, lr);

                model.async_start_eval();
                tracing::debug!("Batch recorded in {:.2?}", start.elapsed());

                result_tx.send((labels, probs, loss)).unwrap();

                num_batches += 1;

                if only_one_iter {
                    return;
                }
            }

            lr *= lr_gamma;

            drop(result_tx);
            thread.join().unwrap();

            let time = epoch_start.elapsed();
            tracing::info!("Epoch cost average of {:.2?}/batch", time / num_batches);
        }
    });

    if only_one_iter {
        return;
    }

    tracing::info!("Training complete, computing validation accuracy...");

    let mut num_correct = 0;
    let validation_batch_size = 2_500;
    for batch in validation_data.chunks(validation_batch_size) {
        let tensor_batch = Batch::new(batch, device);
        let output = model
            .forward(tensor_batch.images, ForwardMode::Inference)
            .to_data_type(DataType::F32)
            .log_softmax()
            .exp()
            .detach();
        let output = output.to_vec::<f32>();
        assert_eq!(output.len(), batch.len() * 10);
        for (item, probs) in batch.iter().zip(output.chunks_exact(10)) {
            let prediction = probs
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0;
            if prediction == item.label {
                num_correct += 1;
            }
        }
    }

    tracing::info!(
        "Validation accuracy: {:.2}%",
        num_correct as f64 / validation_data.len() as f64 * 100.0
    );
}
