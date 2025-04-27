use crate::{DataType, Device, Param, Tensor};
use helium::initializer::Initializer;
use helium_ir::opgraph::op::precision::Precision;
use helium_macros::Module;
use rand::Rng;

/// Linear / feed-forward layer. Output is a linear
/// combination of the inputs, plus optional bias.
#[derive(Clone, Module)]
pub struct Linear {
    weights: Param<2>,
    bias: Option<Param<1>>,
}

impl Linear {
    pub fn new(settings: LinearSettings, rng: &mut impl Rng, device: Device) -> Self {
        let weights = settings.weight_initializer.initialize(
            [settings.in_features, settings.out_features],
            settings.data_type,
            rng,
            device,
        );
        let bias = if settings.bias {
            Some(Tensor::ones(
                [settings.out_features],
                settings.data_type,
                device,
            ))
        } else {
            None
        };

        weights.async_start_eval();
        if let Some(bias) = &bias {
            bias.async_start_eval();
        }

        Self {
            weights: Param::new(weights),
            bias: bias.map(Param::new),
        }
    }

    pub fn forward(&self, x: &Tensor<2>) -> Tensor<2> {
        let precision = match self.weights.value().data_type() {
            DataType::Bf16 => Precision::MulBf16AccumF32,
            DataType::F16 => Precision::MulF16AccumF32,
            DataType::F32 => Precision::MulTf32AccumF32,
            _ => unreachable!(),
        };

        let mut x = x.matmul(self.weights.value().to_data_type(x.data_type()), precision);
        if let Some(bias) = &self.bias {
            x = &x + bias.value().broadcast_to(x.shape());
        }
        x
    }
}

pub struct LinearSettings {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
    pub data_type: DataType,
    pub weight_initializer: Initializer,
}

impl Default for LinearSettings {
    fn default() -> Self {
        Self {
            in_features: 0,
            out_features: 0,
            bias: true,
            data_type: DataType::F32,
            weight_initializer: Initializer::KaimingNormal,
        }
    }
}
