use crate::{DataType, Device, Param, Tensor, conv::Conv2dParams};
use helium::initializer::Initializer;
use helium_macros::Module;
use rand::Rng;

/// 2D convolutional layer, with optional per-channel bias.
///
/// Inputs are expected in NHWC layout, and kernels
/// in KRSC, where:
/// `N` = batch size
/// `H` = height
/// `W` = width
/// `C` = input channel count
/// `K` = output channel count
/// `R` = kernel height
/// `S` = kernel width
#[derive(Module, Clone)]
pub struct Conv2d {
    kernel: Param<4>,
    bias: Option<Param<1>>,
    #[module(config)]
    params: Conv2dParams,
}

impl Conv2d {
    pub fn new(settings: Conv2dSettings, rng: &mut impl Rng, device: Device) -> Self {
        settings.params.validate();
        let kernel_shape = [
            settings.params.out_channels,
            settings.params.kernel_size[0],
            settings.params.kernel_size[1],
            settings.params.in_channels,
        ];
        let kernel =
            settings
                .kernel_initializer
                .initialize(kernel_shape, settings.data_type, rng, device);

        let bias = if settings.bias {
            Some(Tensor::zeros(
                [settings.params.out_channels],
                settings.data_type,
                device,
            ))
        } else {
            None
        };

        kernel.async_start_eval();
        if let Some(bias) = &bias {
            bias.async_start_eval();
        }

        Self {
            kernel: Param::new(kernel),
            bias: bias.map(Param::new),
            params: settings.params,
        }
    }

    pub fn forward(&self, x: &Tensor<4>) -> Tensor<4> {
        let mut x = x.conv2d(self.kernel.value().to_data_type(x.data_type()), self.params);
        if let Some(bias) = &self.bias {
            x = &x + bias.value().broadcast_to(x.shape());
        }
        x
    }

    pub fn params(&self) -> Conv2dParams {
        self.params
    }
}

pub struct Conv2dSettings {
    pub params: Conv2dParams,
    pub kernel_initializer: Initializer,
    pub bias: bool,
    pub data_type: DataType,
}

impl Default for Conv2dSettings {
    fn default() -> Self {
        Self {
            params: Conv2dParams::default(),
            kernel_initializer: Initializer::KaimingNormal,
            bias: true,
            data_type: DataType::F32,
        }
    }
}
