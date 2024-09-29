use crate::Tensor;
use std::any::Any;

/// Tensor with runtime number of dimensions.
pub struct DynTensor {
    tensor: Box<dyn Any + Send + Sync>,
    clone: fn(&(dyn Any + Send + Sync)) -> Box<dyn Any + Send + Sync>,
}

impl DynTensor {
    pub fn new<const D: usize>(tensor: Tensor<D>) -> Self {
        fn clone<const D: usize>(tensor: &(dyn Any + Send + Sync)) -> Box<dyn Any + Send + Sync> {
            Box::new(tensor.downcast_ref::<Tensor<D>>().unwrap().clone())
        }
        Self {
            tensor: Box::new(tensor),
            clone: clone::<D>,
        }
    }

    pub fn get<const D: usize>(&self) -> &Tensor<D> {
        self.tensor
            .downcast_ref()
            .expect("wrong number of tensor dimensions")
    }

    pub fn into_inner<const D: usize>(self) -> Tensor<D> {
        *self
            .tensor
            .downcast()
            .ok()
            .expect("wrong number of tensor dimensions")
    }
}

impl Clone for DynTensor {
    fn clone(&self) -> Self {
        Self {
            tensor: (self.clone)(&*self.tensor),
            clone: self.clone,
        }
    }
}
