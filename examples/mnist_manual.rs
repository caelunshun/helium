use helium::{AdTensor, Device, Gradients, Param, Tensor};
use mnist::MnistBuilder;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

struct Model {
    layer1: Param<2>,
    //bias1: Param<1>,
    layer2: Param<2>,
    //bias2: Param<1>,
    layer3: Param<2>,
    //bias3: Param<1>,
}

impl Model {
    pub fn with_random_weights(rng: &mut impl Rng, device: Device) -> Self {
        Self {
            layer1: init_xavier(28 * 28, 256, rng, device).into(),
            layer2: init_xavier(256, 128, rng, device).into(),
            layer3: init_xavier(128, 10, rng, device).into(),
        }
    }

    pub fn forward(&self, mut x: AdTensor<2>) -> AdTensor<2> {
        x = x.transpose();
        for (i, layer) in [&self.layer1, &self.layer2, &self.layer3]
            .into_iter()
            .enumerate()
        {
            x = AdTensor::from(layer.clone()).matmul(x);
            if i != 2 {
                x = x.sigmoid();
            }
        }
        x.transpose()
    }

    pub fn update_weights(&mut self, mut grads: Gradients, learning_rate: f32) {
        for layer in [&mut self.layer1, &mut self.layer2, &mut self.layer3] {
            let grad = grads.remove::<2>(layer.id());
            layer.set_value(layer.value().clone() - grad * learning_rate);
        }
    }
}

fn softmax(x: AdTensor<2>) -> AdTensor<2> {
    let x = x.exp();
    let denom = x.clone().reduce_sum::<2>(1).broadcast_to(x.shape());
    x / denom
}

fn cross_entropy_loss(logits: AdTensor<2>, targets: AdTensor<2>) -> AdTensor<1> {
    (softmax(logits).log() * targets).reduce_mean::<1>(2) * -1.0
}

fn init_zeros(len: usize, device: Device) -> Tensor<1> {
    Tensor::from_vec(vec![0.0f32; len], [len], device)
}

fn init_xavier(inputs: usize, outputs: usize, rng: &mut impl Rng, device: Device) -> Tensor<2> {
    let x = (inputs as f32).sqrt().recip();
    let weights = (0..inputs * outputs)
        .map(|_| rng.gen_range(-x..=x))
        .collect::<Vec<_>>();
    Tensor::from_vec(weights, [outputs, inputs], device)
}

#[derive(Debug, Clone)]
struct Item {
    image: Vec<f32>,
    label_one_hot: Vec<f32>,
}

fn main() {
    let device = Device::Cuda(0);
    let mut rng = Pcg64Mcg::seed_from_u64(666);
    let mut model = Model::with_random_weights(&mut rng, device);

    let mnist = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(60_000)
        .validation_set_length(0)
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

    let num_epochs = 5;
    let lr = 1e-2;
    let batch_size = 32;

    for epoch in 0..num_epochs {
        items.shuffle(&mut rng);

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
            let input = AdTensor::new(Tensor::<2>::from_vec(input, [batch_size, 28 * 28], device));
            let labels = AdTensor::new(Tensor::<2>::from_vec(labels, [batch_size, 10], device));

            let logits = model.forward(input);
            let loss = cross_entropy_loss(logits, labels);

            let grads = loss.clone().backward();
            model.update_weights(grads, lr);

            let loss_scalar = loss.into_value().into_scalar::<f32>();
            dbg!(loss_scalar);
        }
    }
}
