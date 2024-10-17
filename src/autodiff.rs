use crate::Tensor;
use std::ops::{Add, Mul, Sub};

mod gradients;
mod param;
mod tape;

use ahash::AHashMap;
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

    pub fn shape(&self) -> [usize; D] {
        self.tensor.shape()
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

    pub fn exp(self) -> Self {
        let result = self.tensor.exp();
        let result2 = result.clone();
        let tape = self
            .tape
            .append_unary(move |flow: Tensor<D>| result2.clone() * flow);
        Self {
            tape,
            tensor: result,
        }
    }

    pub fn log(self) -> Self {
        let input = self.tensor.clone();
        let tape = self
            .tape
            .append_unary(move |flow: Tensor<D>| input.clone().recip() * flow);
        Self {
            tape,
            tensor: self.tensor.log(),
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

    pub fn swap_dims(self, axis_a: usize, axis_b: usize) -> Self {
        let tape = self
            .tape
            .append_unary(move |flow: Tensor<D>| flow.swap_dims(axis_b, axis_a));
        Self {
            tensor: self.tensor.swap_dims(axis_a, axis_b),
            tape,
        }
    }

    pub fn broadcast_to<const D2: usize>(self, new_shape: [usize; D2]) -> AdTensor<D2> {
        let old_shape = self.tensor.shape();

        let mut broadcast_axes = Vec::new();
        for i in 0..D {
            let j = (D2 - D) + i;
            if new_shape[j] != 1 && old_shape[i] == 1 {
                broadcast_axes.push(j);
            }
        }
        broadcast_axes.extend(0..(D2 - D));

        // compute axis index mapping to shift broadcast axes to bottom
        let mut axis_mapping: AHashMap<usize, usize> = AHashMap::new();
        for (k, axis) in broadcast_axes.iter().copied().enumerate() {
            axis_mapping.insert(axis, k);
        }
        let mut offset = broadcast_axes.len();
        for axis in 0..D2 {
            if !broadcast_axes.contains(&axis) {
                axis_mapping.insert(axis, offset);
                offset += 1;
            }
        }
        axis_mapping.values_mut().for_each(|x| *x = D2 - *x - 1);

        let inv_axis_mapping: AHashMap<usize, usize> =
            axis_mapping.iter().map(|(&a, &b)| (b, a)).collect();

        let tape = self.tape.append_unary(move |mut flow: Tensor<D2>| {
            for (a, b) in apply_permutation_via_swaps(D2, &|x| axis_mapping[&x]) {
                flow = flow.swap_dims(a, b);
            }

            let flow = flow.reduce_sum::<D>(broadcast_axes.len() as u32);
            let mut temp_shape = flow.shape().to_vec();
            while temp_shape.len() < D2 {
                temp_shape.push(1);
            }
            let mut flow: Tensor<D2> = flow.reshape(temp_shape.try_into().unwrap());

            for (a, b) in apply_permutation_via_swaps(D2, &|x| inv_axis_mapping[&x]) {
                flow = flow.swap_dims(a, b);
            }

            flow.reshape(old_shape)
        });

        AdTensor {
            tape,
            tensor: self.tensor.broadcast_to(new_shape),
        }
    }

    pub fn reduce_sum<const D2: usize>(self, depth: u32) -> AdTensor<D2> {
        let shape = self.shape();
        let result: Tensor<D2> = self.tensor.reduce_sum(depth);
        let result_shape = result.shape();

        let tape = self.tape.append_unary(move |flow: Tensor<D2>| {
            let mut new_shape = [1usize; D];
            new_shape[..D2].copy_from_slice(&result_shape);
            flow.reshape(new_shape).broadcast_to(shape)
        });

        AdTensor {
            tensor: result,
            tape,
        }
    }

    pub fn reduce_mean<const D2: usize>(self, depth: u32) -> AdTensor<D2> {
        let shape = self.shape();
        let stride = shape[shape.len() - depth as usize..]
            .iter()
            .copied()
            .product::<usize>();
        let result: Tensor<D2> = self.tensor.reduce_sum(depth);
        let result_shape = result.shape();

        let tape = self.tape.append_unary(move |flow: Tensor<D2>| {
            let mut new_shape = [1usize; D];
            new_shape[..D2].copy_from_slice(&result_shape);
            flow.reshape(new_shape).broadcast_to(shape) * (stride as f32).recip()
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

/// extremely rare practical application of group theory
fn apply_permutation_via_swaps(n: usize, perm: &dyn Fn(usize) -> usize) -> Vec<(usize, usize)> {
    let mut swaps = Vec::new();
    let mut perm = (0..n).map(perm).collect::<Vec<_>>();
    for i in 0..n {
        if perm[i] == i {
            continue;
        }
        for j in (i + 1)..n {
            if perm[j] == i {
                swaps.push((i, j));
                perm.swap(i, j);
                break;
            }
        }
    }
    swaps
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{seq::SliceRandom, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    #[test]
    fn test_apply_permutation_via_swaps() {
        fn check(n: usize, permutation: &dyn Fn(usize) -> usize) {
            let swaps = apply_permutation_via_swaps(n, permutation);
            let mut vals = (0..n).collect::<Vec<usize>>();
            for (a, b) in swaps {
                vals.swap(a, b);
            }

            vals.iter()
                .copied()
                .enumerate()
                .for_each(|(i, j)| assert_eq!(i, permutation(j)));
        }

        fn cyclic(n: usize) -> impl Fn(usize) -> usize {
            move |x| (x + 1) % n
        }

        fn random(n: usize) -> impl Fn(usize) -> usize {
            let mut perm: Vec<_> = (0..n).collect();
            perm.shuffle(&mut Pcg64Mcg::seed_from_u64(500));

            move |x| perm[x]
        }

        check(2, &cyclic(2));
        check(3, &cyclic(3));
        check(10, &cyclic(10));
        check(10, &random(10));
        check(100, &random(100));
    }
}
