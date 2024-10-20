use helium::{Device, Gradients, Param, Tensor};
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

    pub fn forward(&self, mut x: Tensor<2>) -> Tensor<2> {
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i != self.layers.len() - 1 {
                x = x.sigmoid();
            }
        }
        x
    }

    pub fn update_weights(&mut self, grads: Gradients, learning_rate: f32) {
        for layer in &mut self.layers {
            layer.update_weights(&grads, learning_rate);
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

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let [batch_size, _] = x.shape();
        let [output_size] = self.bias.value().shape();
        x.matmul(self.weights.value()) + self.bias.value().broadcast_to([batch_size, output_size])
    }

    pub fn update_weights(&mut self, grads: &Gradients, learning_rate: f32) {
        let weight_grad = grads.get::<2>(self.weights.id());
        let bias_grad = grads.get::<1>(self.bias.id());

        self.weights
            .set_value(self.weights.value().clone() - weight_grad * learning_rate);
        self.bias
            .set_value(self.bias.value().clone() - bias_grad * learning_rate);
    }
}

fn softmax(x: Tensor<2>) -> Tensor<2> {
    let x = x.exp();
    let denom = x.clone().reduce_sum::<2>(1).broadcast_to(x.shape());
    x / denom
}

#[expect(unused)]
fn cross_entropy_loss(logits: Tensor<2>, targets: Tensor<2>) -> Tensor<1> {
    ((softmax(logits) + 1e-6).log() * targets).reduce_mean::<1>(2) * -1.0
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

    let num_epochs = 10;
    let lr = 1e-2;
    let batch_size = 1024;

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
            let input = Tensor::<2>::from_vec(input, [batch_size, 28 * 28], device);
            let labels = Tensor::<2>::from_vec(labels, [batch_size, 10], device);

            let logits = model.forward(input).sigmoid();
            let loss = (logits - labels).pow_scalar(2.0).reduce_mean::<1>(2);

            let grads = loss.clone().backward();
            model.update_weights(grads, lr);

            let loss_scalar = loss.to_scalar::<f32>();
            println!("Training batch loss: {loss_scalar:.3}");
        }
    }

    // Validation
    let inputs: Vec<f32> = validation_items
        .iter()
        .flat_map(|item| item.image.as_slice())
        .copied()
        .collect();
    let labels: Vec<f32> = validation_items
        .iter()
        .flat_map(|item| item.label_one_hot.as_slice())
        .copied()
        .collect();
    let inputs = Tensor::from_vec(inputs, [validation_items.len(), 28 * 28], device);

    let outputs = model.forward(inputs.into()).sigmoid().to_vec::<f32>();

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
