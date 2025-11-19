use helium::{
    conv::Conv2dParams,
    initializer::Initializer,
    modules::{batch_norm::ForwardMode, conv::Conv2dSettings, BatchNorm2d, Conv2d},
    Device, Module, Tensor,
};
use rand::prelude::*;

pub const NUM_CLASSES: usize = 1000;

/// CNN mostly based on ResnetV2 from
/// [this paper](https://arxiv.org/abs/1603.05027 by He et al.
#[derive(Clone, Module)]
pub struct Model {
    head: Conv2d,
    segments: Vec<Segment>,
    tail_bn: BatchNorm2d,
    tail: Conv2d,
}

impl Model {
    pub fn new(rng: &mut impl Rng, device: Device) -> Self {
        let head = Conv2d::new(
            Conv2dSettings {
                params: Conv2dParams {
                    kernel_size: [7, 7],
                    stride: [2, 2],
                    in_channels: 3,
                    out_channels: 32,
                    ..Default::default()
                },
                kernel_initializer: Initializer::KaimingNormal,
                ..Default::default()
            },
            rng,
            device,
        );
        let tail_bn = BatchNorm2d::new(512, device);
        let tail = Conv2d::new(
            Conv2dSettings {
                params: Conv2dParams {
                    kernel_size: [1, 1],
                    in_channels: 512,
                    out_channels: NUM_CLASSES,
                    ..Default::default()
                },
                kernel_initializer: Initializer::KaimingNormal,
                ..Default::default()
            },
            rng,
            device,
        );
        let segments = vec![
            Segment::new(32, 1, rng, device),
            Segment::new(64, 1, rng, device),
            Segment::new(128, 1, rng, device),
            Segment::new(256, 1, rng, device),
        ];

        Self {
            head,
            segments,
            tail_bn,
            tail,
        }
    }

    pub fn forward(&mut self, x: &Tensor<4>, mode: ForwardMode) -> Tensor<2> {
        let mut x = self.head.forward(x);
        for segment in &mut self.segments {
            x = segment.forward(&x, mode);
        }
        x = self.tail_bn.forward(&x, mode);
        x = x.relu();
        x = self.tail.forward(&x);

        // Global average pooling
        let [batch_size, ..] = x.shape();
        x.swap_dims(2, 3)
            .swap_dims(1, 2)
            .reduce_mean::<3>(2)
            .reshape([batch_size, NUM_CLASSES])
    }
}

#[derive(Clone, Module)]
struct Segment {
    downsample: DownsampleBlock,
    basic: Vec<BasicBlock>,
}

impl Segment {
    pub fn new(in_channels: usize, basic_count: usize, rng: &mut impl Rng, device: Device) -> Self {
        Self {
            downsample: DownsampleBlock::new(in_channels, rng, device),
            basic: (0..basic_count)
                .map(|_| BasicBlock::new(in_channels * 2, rng, device))
                .collect(),
        }
    }

    pub fn forward(&mut self, x: &Tensor<4>, mode: ForwardMode) -> Tensor<4> {
        let mut x = self.downsample.forward(x, mode);
        for basic in &mut self.basic {
            x = basic.forward(&x, mode);
        }
        x
    }
}

#[derive(Clone, Module)]
struct BasicBlock {
    bn1: BatchNorm2d,
    conv1: Conv2d,
    bn2: BatchNorm2d,
    conv2: Conv2d,
}

impl BasicBlock {
    pub fn new(channels: usize, rng: &mut impl Rng, device: Device) -> Self {
        Self {
            bn1: BatchNorm2d::new(channels, device),
            conv1: Conv2d::new(
                Conv2dSettings {
                    params: Conv2dParams {
                        kernel_size: [3, 3],
                        in_channels: channels,
                        out_channels: channels,
                        ..Default::default()
                    },
                    kernel_initializer: Initializer::KaimingNormal,
                    ..Default::default()
                },
                rng,
                device,
            ),
            bn2: BatchNorm2d::new(channels, device),
            conv2: Conv2d::new(
                Conv2dSettings {
                    params: Conv2dParams {
                        kernel_size: [3, 3],
                        in_channels: channels,
                        out_channels: channels,
                        ..Default::default()
                    },
                    kernel_initializer: Initializer::KaimingNormal,
                    ..Default::default()
                },
                rng,
                device,
            ),
        }
    }

    pub fn forward(&mut self, x: &Tensor<4>, mode: ForwardMode) -> Tensor<4> {
        let shortcut = x.clone();
        let mut x = self.bn1.forward(x, mode);
        x = x.relu();
        x = self.conv1.forward(&x);

        x = self.bn2.forward(&x, mode);
        x = x.relu();
        x = self.conv2.forward(&x);

        x + shortcut
    }
}

/// Dimensions /2, channels *2
#[derive(Clone, Module)]
struct DownsampleBlock {
    bn1: BatchNorm2d,
    conv1: Conv2d,
    bn2: BatchNorm2d,
    conv2: Conv2d,
    shortcut_conv: Conv2d,
}

impl DownsampleBlock {
    pub fn new(in_channels: usize, rng: &mut impl Rng, device: Device) -> Self {
        let out_channels = in_channels * 2;
        Self {
            bn1: BatchNorm2d::new(in_channels, device),
            conv1: Conv2d::new(
                Conv2dSettings {
                    params: Conv2dParams {
                        kernel_size: [3, 3],
                        stride: [2, 2],
                        in_channels,
                        out_channels,
                        ..Default::default()
                    },
                    kernel_initializer: Initializer::KaimingNormal,
                    ..Default::default()
                },
                rng,
                device,
            ),
            bn2: BatchNorm2d::new(out_channels, device),
            conv2: Conv2d::new(
                Conv2dSettings {
                    params: Conv2dParams {
                        kernel_size: [3, 3],
                        in_channels: out_channels,
                        out_channels,
                        ..Default::default()
                    },
                    kernel_initializer: Initializer::KaimingNormal,
                    ..Default::default()
                },
                rng,
                device,
            ),
            shortcut_conv: Conv2d::new(
                Conv2dSettings {
                    params: Conv2dParams {
                        kernel_size: [1, 1],
                        stride: [2, 2],
                        in_channels,
                        out_channels,
                        ..Default::default()
                    },
                    kernel_initializer: Initializer::KaimingNormal,
                    ..Default::default()
                },
                rng,
                device,
            ),
        }
    }

    pub fn forward(&mut self, x: &Tensor<4>, mode: ForwardMode) -> Tensor<4> {
        let mut x = self.bn1.forward(x, mode);
        x = x.relu();

        let shortcut = self.shortcut_conv.forward(&x);

        let mut x = self.bn1.forward(&x, mode);
        x = x.relu();
        x = self.conv1.forward(&x);

        x = self.bn2.forward(&x, mode);
        x = x.relu();
        x = self.conv2.forward(&x);

        x + shortcut
    }
}
