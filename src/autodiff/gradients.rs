use crate::{autodiff::param::ParamId, dyn_tensor::DynTensor, Tensor};
use ahash::AHashMap;

#[derive(Default)]
pub struct Gradients {
    grads: AHashMap<ParamId, DynTensor>,
}

impl Gradients {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<const D: usize>(&mut self, param: ParamId, grad: Tensor<D>) {
        self.insert_dyn(param, DynTensor::new(grad));
    }

    pub fn insert_dyn(&mut self, param: ParamId, grad: DynTensor) {
        self.grads.insert(param, grad);
    }

    pub fn get<const D: usize>(&self, param: ParamId) -> &Tensor<D> {
        self.grads.get(&param).expect("missing gradient").get()
    }

    pub fn remove<const D: usize>(&mut self, param: ParamId) -> Tensor<D> {
        self.grads
            .remove(&param)
            .expect("missing gradient")
            .into_inner()
    }
}
