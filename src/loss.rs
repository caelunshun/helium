//! Common loss functions.

use crate::Tensor;

pub fn cross_entropy_loss(logits: Tensor<2>, targets: Tensor<2>) -> Tensor<1> {
    let [batch_size, ..] = logits.shape();
    -(logits.log_softmax() * targets).reduce_sum::<1>(2) / batch_size as f32
}
