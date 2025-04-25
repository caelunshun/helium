//! Common loss functions.

use crate::Tensor;

pub fn cross_entropy_loss(logits: Tensor<2>, targets: Tensor<2>) -> Tensor<1> {
    let [batch_size, ..] = logits.shape();
    -(logits.log_softmax() * targets).reduce_sum::<1>(2) / batch_size as f32
}

pub fn binary_cross_entropy_loss(predictions: &Tensor<2>, targets: &Tensor<2>) -> Tensor<1> {
    -(targets * clamped_log(predictions) + (-targets + 1.0) * clamped_log(&(-predictions + 1.0)))
        .reduce_mean(2)
}

/// Logarithm clamped to be at least -100.
/// See rationale in https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
fn clamped_log<const D: usize>(x: &Tensor<D>) -> Tensor<D> {
    x.log()
        .max(Tensor::from_constant(-100.0f32, x.shape(), x.device()))
}
