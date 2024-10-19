use crate::{
    raw_tensor::RawTensor,
    tensor::{param::ParamId, Tensor},
};
use ahash::AHashMap;

#[derive(Default)]
pub struct Gradients {
    grads: AHashMap<ParamId, RawTensor>,
}

impl Gradients {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<const D: usize>(&mut self, param: ParamId, grad: Tensor<D>) {
        self.insert_raw(param, grad.raw);
    }

    pub(crate) fn insert_raw(&mut self, param: ParamId, grad: RawTensor) {
        self.grads.insert(param, grad);
    }

    pub fn get<const D: usize>(&self, param: ParamId) -> Tensor<D> {
        Tensor::from_raw(self.grads.get(&param).expect("missing gradient").clone())
    }
}
