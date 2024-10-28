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
use helium::{conv::Conv2dSettings, DataType, Device, Gradients, Param, Tensor};
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

struct Model {
    layers: Vec<Conv>,
    batchnorms: Vec<BatchNorm>,
}

impl Model {
    pub fn new(rng: &mut impl Rng, device: Device) -> Self {
        let layers = vec![
            Conv::new(
                Conv2dSettings {
                    kernel_size: [3, 3],
                    in_channels: 3,
                    out_channels: 64,
                    ..Default::default()
                },
                rng,
                device,
            ),
            Conv::new(
                Conv2dSettings {
                    kernel_size: [3, 3],
                    stride: [2, 2],
                    in_channels: 64,
                    out_channels: 128,
                    ..Default::default()
                },
                rng,
                device,
            ),
            Conv::new(
                Conv2dSettings {
                    kernel_size: [3, 3],
                    stride: [2, 2],
                    in_channels: 128,
                    out_channels: 256,
                    ..Default::default()
                },
                rng,
                device,
            ),
            Conv::new(
                Conv2dSettings {
                    kernel_size: [3, 3],
                    stride: [2, 2],
                    in_channels: 256,
                    out_channels: 256,
                    ..Default::default()
                },
                rng,
                device,
            ),
            Conv::new(
                Conv2dSettings {
                    kernel_size: [3, 3],
                    stride: [1, 1],
                    in_channels: 256,
                    out_channels: 10,
                    ..Default::default()
                },
                rng,
                device,
            ),
        ];
        let batchnorms = layers
            .iter()
            .take(layers.len() - 1)
            .map(|layer| BatchNorm::new(layer.settings.out_channels, device))
            .collect();
        Self { layers, batchnorms }
    }

    pub fn forward(&mut self, mut x: Tensor<4>, train: bool) -> Tensor<2> {
        let [batch_size, ..] = x.shape();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i != self.layers.len() - 1 {
                x = self.batchnorms[i].forward(x, train);
                x = x.relu();
            }
        }
        // Global average pooling
        x.swap_dims(2, 3)
            .swap_dims(1, 2)
            .reduce_mean::<3>(2)
            .reshape([batch_size, 10])
    }

    pub fn update_weights(&mut self, grads: &Gradients, learning_rate: f32) {
        for layer in &mut self.layers {
            layer.update_weights(grads, learning_rate);
        }
        for bn in &mut self.batchnorms {
            bn.update_weights(grads, learning_rate);
        }
    }
}

struct Conv {
    settings: Conv2dSettings,
    kernel: Param<4>,
}

impl Conv {
    pub fn new(settings: Conv2dSettings, rng: &mut impl Rng, device: Device) -> Self {
        Self {
            settings,
            kernel: Param::new(init_kaiming(
                settings.kernel_size,
                settings.in_channels,
                settings.out_channels,
                rng,
                device,
            )),
        }
    }

    pub fn forward(&self, mut x: Tensor<4>) -> Tensor<4> {
        x = x.conv2d(
            self.kernel.value().to_data_type(x.data_type()),
            self.settings,
        );
        x
    }

    pub fn update_weights(&mut self, grads: &Gradients, learning_rate: f32) {
        let id = self.kernel.id();
        self.kernel
            .update(|kernel| kernel - grads.get::<4>(id) * learning_rate);
    }
}

struct BatchNorm {
    running_mean: Tensor<1>,
    running_variance: Tensor<1>,
    scale: Param<1>,
    bias: Param<1>,
    num_channels: usize,
}

impl BatchNorm {
    pub fn new(num_channels: usize, device: Device) -> Self {
        let bn = Self {
            running_mean: Tensor::from_scalar(0.0f32, device).broadcast_to([num_channels]),
            running_variance: Tensor::from_scalar(1.0f32, device).broadcast_to([num_channels]),
            scale: Tensor::from_scalar(1.0f32, device)
                .broadcast_to([num_channels])
                .into(),
            bias: Tensor::from_scalar(0.0f32, device)
                .broadcast_to([num_channels])
                .into(),
            num_channels,
        };
        bn.scale.value().async_start_eval();
        bn.bias.value().async_start_eval();
        bn.running_mean.async_start_eval();
        bn.running_variance.async_start_eval();
        bn
    }

    pub fn forward(&mut self, mut x: Tensor<4>, train: bool) -> Tensor<4> {
        let dtype = x.data_type();
        let (mean, variance) = if train {
            // `x` has NHWC layout, but we want to compute
            // statistics over NHW. Swap N and C to solve.
            let xt = x.swap_dims(0, 3);

            let mean = xt.reduce_mean::<2>(3).reshape([self.num_channels]);
            let mean_square = xt
                .pow_scalar(2.0)
                .reduce_mean::<2>(3)
                .reshape([self.num_channels]);
            let variance = mean_square - mean.pow_scalar(2.0);

            self.running_mean = (&self.running_mean * 0.9 + &mean * 0.1)
                .detach()
                .to_data_type(DataType::F32);
            self.running_variance = (&self.running_variance * 0.9 + &variance * 0.1)
                .detach()
                .to_data_type(DataType::F32);

            (mean, variance)
        } else {
            (
                self.running_mean.to_data_type(dtype),
                self.running_variance.to_data_type(dtype),
            )
        };

        let mean_broadcast = mean.broadcast_to(x.shape());
        let variance_broadcast = variance.broadcast_to(x.shape());

        x = (x - mean_broadcast) / (variance_broadcast + 1e-3).sqrt();

        let scale_broadcast = self.scale.value().broadcast_to(x.shape());
        let bias_broadcast = self.bias.value().broadcast_to(x.shape());

        (x * scale_broadcast + bias_broadcast).to_data_type(dtype)
    }

    pub fn update_weights(&mut self, grads: &Gradients, learning_rate: f32) {
        let id = self.scale.id();
        self.scale
            .update(|scale| scale - grads.get::<1>(id) * learning_rate);
        let id = self.bias.id();
        self.bias
            .update(|bias| bias - grads.get::<1>(id) * learning_rate);
    }
}

fn init_kaiming(
    kernel_size: [usize; 2],
    in_channels: usize,
    out_channels: usize,
    rng: &mut impl Rng,
    device: Device,
) -> Tensor<4> {
    let fan_out = (kernel_size[0] * kernel_size[1] * out_channels) as f64;
    let stdev = (2.0 / fan_out).sqrt() as f32;
    let dist = Normal::new(0.0, stdev).unwrap();

    let num_elements = kernel_size[0] * kernel_size[1] * in_channels * out_channels;
    let data: Vec<f32> = (0..num_elements).map(|_| dist.sample(rng)).collect();
    Tensor::from_vec(
        data,
        [out_channels, kernel_size[0], kernel_size[1], in_channels],
        device,
    )
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
    let [batch_size, ..] = logits.shape();
    -(log_softmax(logits) * targets).reduce_sum::<1>(2) / batch_size as f32
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
            images: Tensor::from_vec(
                images,
                [items.len(), IMAGE_DIM, IMAGE_DIM, IMAGE_CHANNELS],
                device,
            ),
            labels: Tensor::from_vec(labels_one_hot, [items.len(), 10], device),
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
    let noise_distr = Normal::new(0.0f32, rng.gen_range(0.001..0.3)).unwrap();
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
        tracing::error!("CIFAR-10 data not downloaded. Follow instructions at top of cifar10_manual.rs to download data.");
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

    let mut lr = 5e-1;
    let lr_gamma = 0.99;
    let num_epochs = 200;
    let batch_size = 1024;

    // For profiling.
    let only_one_iter = false;

    training_data.shuffle(&mut rng);

    thread::scope(|s| {
        for epoch in 0..num_epochs {
            tracing::info!("Starting epoch {epoch} with learning rate {lr}");

            let (batch_tx, batch_rx) = flume::bounded(16);
            let training_data = &training_data;
            let mut reg_rng = Pcg64Mcg::from_rng(&mut rng).unwrap();
            s.spawn(move || {
                for items in training_data.chunks_exact(batch_size) {
                    let mut regularized_items = items.to_vec();
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
                    let labels = items.iter().map(|item| item.label).collect::<Vec<_>>();
                    batch_tx.send((batch, labels)).ok();
                }
            });
            for (batch, labels) in batch_rx {
                let logits = model
                    .forward(batch.images, true)
                    .to_data_type(DataType::F32);
                let loss = cross_entropy_loss(logits.clone(), batch.labels);
                let probs = log_softmax(logits).exp();
                loss.async_start_eval();
                let grads = loss.backward();
                model.update_weights(&grads, lr);
                model.layers[0].kernel.value().async_start_eval();

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

                if only_one_iter {
                    return;
                }
            }

            lr *= lr_gamma;
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
        let output = log_softmax(
            model
                .forward(tensor_batch.images, false)
                .to_data_type(DataType::F32),
        )
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
