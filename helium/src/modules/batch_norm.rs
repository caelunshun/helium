use crate::{DataType, Device, Param, Tensor};
use helium_macros::Module;

/// Batch normalization in two dimensions, computing
/// batch statistics per channel.
#[derive(Module, Clone)]
pub struct BatchNorm2d {
    running_mean: Tensor<1>,
    running_variance: Tensor<1>,
    scale: Param<1>,
    bias: Param<1>,
    #[module(config)]
    num_channels: usize,
}

impl BatchNorm2d {
    pub fn new(num_channels: usize, device: Device) -> Self {
        let bn = Self {
            running_mean: Tensor::zeros([num_channels], DataType::F32, device),
            running_variance: Tensor::ones([num_channels], DataType::F32, device),
            scale: Tensor::ones([num_channels], DataType::F32, device).into(),
            bias: Tensor::zeros([num_channels], DataType::F32, device).into(),
            num_channels,
        };
        bn.scale.value().async_start_eval();
        bn.bias.value().async_start_eval();
        bn.running_mean.async_start_eval();
        bn.running_variance.async_start_eval();
        bn
    }

    pub fn forward(&mut self, x: &Tensor<4>, mode: ForwardMode) -> Tensor<4> {
        let dtype = x.data_type();
        let (mean, variance) = match mode {
            ForwardMode::Train => {
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
            }
            ForwardMode::Inference => (
                self.running_mean.to_data_type(dtype),
                self.running_variance.to_data_type(dtype),
            ),
        };

        let mean_broadcast = mean.broadcast_to(x.shape());
        let variance_broadcast = variance.broadcast_to(x.shape());

        let x = (x - mean_broadcast) / (variance_broadcast + 1e-3).sqrt();

        let scale_broadcast = self.scale.value().broadcast_to(x.shape());
        let bias_broadcast = self.bias.value().broadcast_to(x.shape());

        (x * scale_broadcast + bias_broadcast).to_data_type(dtype)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ForwardMode {
    Train,
    Inference,
}
