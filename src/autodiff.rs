use crate::Tensor;
use std::ops::{Add, Mul};

mod gradients;
mod param;
mod tape;

pub use gradients::Gradients;
pub use param::{Param, ParamId};
pub use tape::Tape;

/// Tensor with autodiff.
pub struct AdTensor<const D: usize> {
    tensor: Tensor<D>,
    tape: Tape,
}

impl<const D: usize> AdTensor<D> {
    pub fn new(tensor: Tensor<D>) -> Self {
        let tape = Tape::new_constant();
        Self { tensor, tape }
    }

    pub fn value(&self) -> &Tensor<D> {
        &self.tensor
    }

    pub fn into_value(self) -> Tensor<D> {
        self.tensor
    }

    pub fn pow_scalar(self, power: f32) -> Self {
        let input = self.tensor.clone();
        let result = self.tensor.pow_scalar(power);

        let tape = self.tape.append_unary(move |flow: Tensor<D>| {
            input.clone().pow_scalar(power - 1.0) * power * flow
        });

        Self {
            tape,
            tensor: result,
        }
    }

    pub fn backward(self) -> Gradients {
        let shape = self.tensor.shape();
        let ones = Tensor::from_vec(
            vec![1.0; shape.iter().copied().product()],
            shape,
            self.tensor.device(),
        );

        self.tape.backward(ones)
    }
}

impl<const D: usize, T> Add<T> for AdTensor<D>
where
    T: Into<Self>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();

        let compute_flow = |flow: Tensor<D>| flow;

        let tape = self
            .tape
            .append_binary(rhs.tape, compute_flow, compute_flow);

        Self {
            tensor: self.tensor + rhs.tensor,
            tape,
        }
    }
}

impl<const D: usize> Add<f32> for AdTensor<D> {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        let tensor = self.tensor + rhs;
        let tape = self.tape.append_unary(|flow: Tensor<D>| flow);
        Self { tensor, tape }
    }
}

impl<const D: usize, T> Mul<T> for AdTensor<D>
where
    T: Into<Self>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();

        let compute_flow1 = {
            let rhs = rhs.tensor.clone();
            move |flow: Tensor<D>| flow * rhs.clone()
        };
        let compute_flow2 = {
            let lhs = self.tensor.clone();
            move |flow: Tensor<D>| flow * lhs.clone()
        };
        let tape = self
            .tape
            .append_binary(rhs.tape, compute_flow1, compute_flow2);

        Self {
            tape,
            tensor: self.tensor * rhs.tensor,
        }
    }
}

impl<const D: usize> Mul<f32> for AdTensor<D> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let tensor = self.tensor * rhs;
        let tape = self.tape.append_unary(move |flow: Tensor<D>| flow * rhs);
        Self { tensor, tape }
    }
}

impl<const D: usize> From<Tensor<D>> for AdTensor<D> {
    fn from(value: Tensor<D>) -> Self {
        Self::new(value)
    }
}

impl<const D: usize> From<Param<D>> for AdTensor<D> {
    fn from(value: Param<D>) -> Self {
        Self {
            tape: Tape::new_param(value.id()),
            tensor: value.into_value(),
        }
    }
}
