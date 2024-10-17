use helium::{AdTensor, Device, Gradients, Param, Tensor};
use mnist::MnistBuilder;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

struct Model {
    layers: [Layer; 3],
}

impl Model {
    pub fn with_random_weights(rng: &mut impl Rng, device: Device) -> Self {
        Self {
            layers: [
                Layer::with_random_weights(28 * 28, 256, rng, device),
                Layer::with_random_weights(256, 128, rng, device),
                Layer::with_random_weights(128, 10, rng, device),
            ],
        }
    }

    pub fn forward(&self, mut x: AdTensor<2>) -> AdTensor<2> {
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i != self.layers.len() - 1 {
                x = x.sigmoid();
            }
        }
        x
    }

    pub fn update_weights(&mut self, mut grads: Gradients, learning_rate: f32) {
        for layer in &mut self.layers {
            layer.update_weights(&mut grads, learning_rate);
        }
    }
}

struct Layer {
    weights: Param<2>,
    bias: Param<1>,
}

impl Layer {
    pub fn with_random_weights(
        inputs: usize,
        outputs: usize,
        rng: &mut impl Rng,
        device: Device,
    ) -> Self {
        Self {
            weights: init_xavier(inputs, outputs, rng, device).into(),
            bias: init_zeros(outputs, device).into(),
        }
    }

    pub fn forward(&self, x: AdTensor<2>) -> AdTensor<2> {
        let [batch_size, _] = x.shape();
        let [output_size] = self.bias.value().shape();
        x.matmul(self.weights.clone().into())
            + AdTensor::from(self.bias.clone()).broadcast_to([batch_size, output_size])
    }

    pub fn update_weights(&mut self, grads: &mut Gradients, learning_rate: f32) {
        let weight_grad = grads.remove::<2>(self.weights.id());
        let bias_grad = grads.remove::<1>(self.bias.id());

        self.weights
            .set_value(self.weights.value().clone() - weight_grad * learning_rate);
        self.bias
            .set_value(self.bias.value().clone() - bias_grad * learning_rate);
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
    Tensor::from_vec(weights, [inputs, outputs], device)
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

    for _epoch in 0..num_epochs {
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