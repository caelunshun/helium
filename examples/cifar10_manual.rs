//! Training on CIFAR-10 image classification task,
//! with convolutional layers.
//!
//! To download data:
//! ```bash
//! mkdir -p data && cd data
//! curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
//! tar -xvf cifar-10-binary.tar.gz
//! ```

use helium::{conv::Conv2dSettings, Device, Gradients, Param, Tensor};
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
}

impl Model {
    pub fn new(rng: &mut impl Rng, device: Device) -> Self {
        Self {
            layers: vec![
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
            ],
        }
    }

    pub fn forward(&self, mut x: Tensor<4>) -> Tensor<2> {
        let [batch_size, ..] = x.shape();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i != self.layers.len() - 1 {
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
    }
}

struct Conv {
    settings: Conv2dSettings,
    kernel: Param<4>,
    bias: Param<1>,
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
            bias: Param::new(init_zeros(settings.out_channels, device)),
        }
    }

    pub fn forward(&self, mut x: Tensor<4>) -> Tensor<4> {
        x = x.conv2d(&self.kernel, self.settings);
        x = &x + self.bias.value().broadcast_to(x.shape());
        x
    }

    pub fn update_weights(&mut self, grads: &Gradients, learning_rate: f32) {
        let id = self.kernel.id();
        self.kernel
            .update(|kernel| kernel - grads.get::<4>(id) * learning_rate);
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

fn init_zeros(len: usize, device: Device) -> Tensor<1> {
    Tensor::from_vec(vec![0.0f32; len], [len], device)
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

struct DataItem {
    // HWC layout
    image: [f32; IMAGE_DIM * IMAGE_DIM * IMAGE_CHANNELS],
    label: usize,
}

struct Batch {
    images: Tensor<4>,
    /// One-hot encoding
    labels: Tensor<2>,
}

impl Batch {
    pub fn new(items: &[DataItem], device: Device) -> Self {
        let images: Vec<f32> = items.iter().flat_map(|item| item.image).collect();
        let labels_one_hot: Vec<f32> = items
            .iter()
            .flat_map(|item| {
                let mut arr = [0.0f32; 10];
                arr[item.label] = 1.0;
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
            image: new_image.map(|x| (x as f32 / 255.0).powf(2.2)),
            label: label_byte[0] as usize,
        });
    }
    items
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

    let mut lr = 1e-1;
    let lr_gamma = 0.975;
    let num_epochs = 100;
    let batch_size = 1024;

    training_data.shuffle(&mut rng);

    thread::scope(|s| {
        for epoch in 0..num_epochs {
            tracing::info!("Starting epoch {epoch} with learning rate {lr}");

            let (batch_tx, batch_rx) = flume::bounded(16);
            let training_data = &training_data;
            s.spawn(move || {
                for items in training_data.chunks_exact(batch_size) {
                    let batch = Batch::new(items, device);
                    batch_tx.send(batch).unwrap();
                }
            });

            for batch in batch_rx {
                let logits = model.forward(batch.images);
                let loss = cross_entropy_loss(logits, batch.labels);
                let grads = loss.backward();
                model.update_weights(&grads, lr);

                tracing::info!(
                    "Epoch {epoch}, training batch loss {:.3}",
                    loss.to_scalar::<f32>()
                );
            }

            lr *= lr_gamma;
        }
    });

    tracing::info!("Training complete, computing validation accuracy...");

    let mut num_correct = 0;
    let validation_batch_size = 2_500;
    for batch in validation_data.chunks(validation_batch_size) {
        let tensor_batch = Batch::new(&batch, device);
        let output = log_softmax(model.forward(tensor_batch.images))
            .exp()
            .detach()
            .to_vec::<f32>();
        for (item, probs) in batch.iter().zip(output.chunks_exact(10)) {
            let class = probs
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0;
            if class == item.label {
                num_correct += 1;
            }
        }
    }

    tracing::info!(
        "Validation accuracy: {:.2}%",
        num_correct as f64 / validation_data.len() as f64 * 100.0
    );
}
