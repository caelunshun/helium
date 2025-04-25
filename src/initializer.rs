use crate::{DataType, Device};
use helium::Tensor;
use rand::Rng;
use rand_distr::Normal;

/// Initializer for weights.
#[derive(Debug, Clone)]
pub enum Initializer {
    Zero,
    One,
    /// Xavier initialization, appropriate for sigmoid activation
    /// functions.
    XavierUniform,
    /// Kaiming He initialization, appropriate for ReLU and ReLU-derived
    /// activation functions.
    KaimingNormal,
}

impl Initializer {
    pub fn initialize<const D: usize>(
        &self,
        shape: [usize; D],
        data_type: DataType,
        rng: &mut impl Rng,
        device: Device,
    ) -> Tensor<D> {
        let size = shape.iter().product::<usize>();

        let (fan_in, fan_out) = match D {
            2 => (shape[0], shape[1]),
            4 => {
                let in_channels = shape[3];
                let out_channels = shape[0];
                let dim = shape[1] * shape[2];
                (dim * in_channels, dim * out_channels)
            }
            _ => panic!("unsupported tensor dimension {D} for initialization"),
        };

        match self {
            Initializer::Zero => Tensor::zeros(shape, data_type, device),
            Initializer::One => Tensor::ones(shape, data_type, device),
            Initializer::XavierUniform => {
                let x = (6.0f64 / (fan_in as f64 + fan_out as f64)).sqrt() as f32;
                let samples: Vec<f32> = (0..size).map(|_| rng.random_range(-x..=x)).collect();
                Tensor::from_slice(samples, shape, device)
            }
            Initializer::KaimingNormal => {
                let stdev = (1.0f64 / (fan_out as f64).sqrt()) as f32;
                let dist = Normal::new(0.0, stdev).unwrap();
                let samples: Vec<f32> = (0..size).map(|_| rng.sample(dist)).collect();
                Tensor::from_slice(samples, shape, device)
            }
        }
    }
}
