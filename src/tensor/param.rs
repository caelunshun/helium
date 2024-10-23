use crate::{tensor::tape::Tape, Tensor};
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
    pub fn new(mut value: Tensor<D>) -> Self {
        let id = ParamId::new();
        value.tape = Some(Tape::new_param(id, value.raw.clone()));
        Self { value, id }
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

    pub fn set_value(&mut self, mut value: Tensor<D>) {
        value.tape = Some(Tape::new_param(self.id, value.raw.clone()));
        self.value = value;
    }

    pub fn update(&mut self, update: impl FnOnce(&Tensor<D>) -> Tensor<D>) {
        let new = update(&self.value);
        self.set_value(new);
    }
}

impl<const D: usize> From<Tensor<D>> for Param<D> {
    fn from(value: Tensor<D>) -> Self {
        Self::new(value)
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
