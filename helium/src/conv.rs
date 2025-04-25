use serde::{Deserialize, Serialize};

/// Convolution parameters.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Conv2dParams {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
    pub padding_mode: PaddingMode,
}

impl Default for Conv2dParams {
    fn default() -> Self {
        Self {
            in_channels: 0,
            out_channels: 0,
            kernel_size: [1, 1],
            stride: [1, 1],
            dilation: [1, 1],
            padding_mode: PaddingMode::Same,
        }
    }
}

impl Conv2dParams {
    pub fn validate(&self) {
        assert!(
            self.in_channels > 0,
            "conv requires nonzero number of input channels"
        );
        assert!(
            self.out_channels > 0,
            "conv requires nonzero number of output channels"
        );
        assert!(
            self.kernel_size[0] > 0 && self.kernel_size[1] > 0,
            "conv filter must have nonzero size"
        );
        assert!(
            self.stride[0] > 0 && self.stride[1] > 0,
            "stride cannot be zero"
        );
        assert!(
            self.dilation[0] > 0 && self.dilation[1] > 0,
            "dilation cannot be zero"
        );
    }

    pub fn compute_output_size(
        &self,
        input_size: [usize; 2],
        kernel_size: [usize; 2],
    ) -> [usize; 2] {
        let mut size = input_size;

        match self.padding_mode {
            PaddingMode::Full => {
                size[0] += kernel_size[0] / 2;
                size[1] += kernel_size[1] / 2;
            }
            PaddingMode::Valid => {
                size[0] = size[0].saturating_sub(kernel_size[0] / 2);
                size[1] = size[1].saturating_sub(kernel_size[1] / 2);
            }
            PaddingMode::Same => {}
        }

        size[0] /= self.stride[0];
        size[1] /= self.stride[1];

        size
    }
}

/// Padding mode for convolution.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PaddingMode {
    Same,
    Valid,
    Full,
}

impl PaddingMode {
    pub fn compute_padding_amount(self, kernel_size: [usize; 2]) -> [usize; 2] {
        match self {
            PaddingMode::Same => [kernel_size[0] / 2, kernel_size[1] / 2],
            PaddingMode::Valid => [0, 0],
            PaddingMode::Full => [kernel_size[0] - 1, kernel_size[1] - 1],
        }
    }
}
