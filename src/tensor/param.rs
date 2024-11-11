use crate::{tensor::tape::Tape, Tensor};
use serde::{Deserialize, Serialize};

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

    pub(crate) fn new_with_id(value: Tensor<D>, id: ParamId) -> Self {
        Self {
            id,
            ..Self::new(value)
        }
    }

    pub fn id(&self) -> ParamId {
        self.id
    }

    pub fn value(&self) -> &Tensor<D> {
        &self.value
    }

    pub(crate) fn value_mut(&mut self) -> &mut Tensor<D> {
        &mut self.value
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ParamId(u128);

impl ParamId {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self(rand::random())
    }
}
