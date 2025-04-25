use approx::assert_ulps_eq;
use helium::{Device, Tensor, loss::binary_cross_entropy_loss};

const DEVICE: Device = Device::Cuda(0);

#[test]
fn bce_loss_basic() {
    let prediction = Tensor::from_constant(0.2, [1, 1], DEVICE);
    let target = Tensor::from_constant(1.0, [1, 1], DEVICE);
    let loss = binary_cross_entropy_loss(&prediction, &target);
    assert_ulps_eq!(loss.to_scalar::<f32>(), -(0.2f32.ln()));
}

#[test]
fn bce_loss_clamp() {
    // See numerical stability concerns in
    // https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    let prediction = Tensor::from_constant(0.0, [1, 1], DEVICE);
    let target = Tensor::from_constant(1.0, [1, 1], DEVICE);
    let loss = binary_cross_entropy_loss(&prediction, &target);
    assert_ulps_eq!(loss.to_scalar::<f32>(), 100.0);
}
