use crate::Tensor;
use std::sync::atomic::AtomicU64;

/// A model parameter tensor.
///
/// Unlike `Tensor`, this is a mutable value.
#[derive(Clone)]
pub struct Param<const D: usize> {
    value: Tensor<D>,
    id: ParamId,
}

impl<const D: usize> Param<D> {
    pub fn new(value: Tensor<D>) -> Self {
        Self {
            value,
            id: ParamId::new(),
        }
    }

    pub fn id(&self) -> ParamId {
        self.id
    }

    pub fn value(&self) -> &Tensor<D> {
        &self.value
    }

    pub fn into_value(self) -> Tensor<D> {
        self.value
    }

    pub fn set_value(&mut self, value: Tensor<D>) {
        self.value = value;
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParamId(u64);

impl ParamId {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        ParamId(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}
