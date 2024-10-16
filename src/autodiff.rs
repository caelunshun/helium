use crate::Tensor;
use std::ops::{Add, Mul, Sub};

mod gradients;
mod param;
mod tape;

pub use gradients::Gradients;
pub use param::{Param, ParamId};
pub use tape::Tape;

/// Tensor with autodiff.
#[derive(Clone)]
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

    pub fn pow(self, power: Self) -> Self {
        let input = self.tensor.clone();
        let power_clone = power.tensor.clone();
        let result = self.tensor.pow(power.tensor);

        let tape = self.tape.append_unary(move |flow: Tensor<D>| {
            input.clone().pow(power_clone.clone() - 1.0) * power_clone.clone() * flow
        });

        Self {
            tape,
            tensor: result,
        }
    }

    pub fn sigmoid(self) -> Self {
        let input = self.tensor.clone();
        let result = self.tensor.sigmoid();

        let tape = self.tape.append_unary(move |flow: Tensor<D>| {
            let exp = input.clone().exp();
            exp.clone() / (exp + 1.0).pow_scalar(2.0) * flow
        });

        Self {
            tape,
            tensor: result,
        }
    }

    pub fn matmul(self, rhs: Self) -> Self {
        let result = self.tensor.clone().matmul(rhs.tensor.clone());

        let a = self.tensor.clone();
        let b = rhs.tensor.clone();

        let tape = self.tape.append_binary(
            rhs.tape,
            move |flow: Tensor<D>| b.clone().matmul(flow.transpose()),
            move |flow: Tensor<D>| a.clone().transpose().matmul(flow),
        );

        Self {
            tape,
            tensor: result,
        }
    }

    pub fn reduce_sum<const D2: usize>(self, depth: u32) -> AdTensor<D2> {
        let input_shape = self.tensor.shape();
        let result: Tensor<D2> = self.tensor.reduce_sum(depth);

        let tape = self.tape.append_unary(move |flow: Tensor<D2>| {
            let mut new_shape = [1usize; D];
            new_shape[..D2].copy_from_slice(&input_shape[..D2]);
            flow.broadcast_to(new_shape)
        });

        AdTensor {
            tensor: result,
            tape,
        }
    }

    pub fn reduce_mean<const D2: usize>(self, depth: u32) -> AdTensor<D2> {
        let input_shape = self.tensor.shape();
        let stride = input_shape[input_shape.len() - depth as usize..]
            .iter()
            .copied()
            .product::<usize>();
        let result: Tensor<D2> = self.tensor.reduce_sum(depth);

        let tape = self.tape.append_unary(move |flow: Tensor<D2>| {
            let mut new_shape = [1usize; D];
            new_shape[..D2].copy_from_slice(&input_shape[..D2]);
            flow.broadcast_to(new_shape) * (stride as f32).recip()
        });

        AdTensor {
            tensor: result,
            tape,
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

impl<const D: usize, T> Sub<T> for AdTensor<D>
where
    T: Into<Self>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();

        let tape =
            self.tape
                .append_binary(rhs.tape, |flow: Tensor<D>| flow, |flow: Tensor<D>| -flow);

        Self {
            tensor: self.tensor - rhs.tensor,
            tape,
        }
    }
}

impl<const D: usize> Sub<f32> for AdTensor<D> {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        let tape = self.tape.append_unary(|flow: Tensor<D>| flow);

        Self {
            tensor: self.tensor - rhs,
            tape,
        }
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
