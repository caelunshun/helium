use crate::{
    DataType, Device, Gradients, Param,
    data_type::{AsDataSlice, DataClass, DataClassTrait, DataTypeConversion, Float, Scalar},
    raw_tensor::RawTensor,
    tensor::tape::Tape,
};
use ahash::AHashMap;
use bytemuck::Pod;
use half::{bf16, f16};
use helium_ir::opgraph::op::{conv::Conv2dParams, precision::Precision};
use pollster::FutureExt;
use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

pub mod gradients;
pub mod param;
pub mod tape;

#[derive(Clone)]
pub struct Tensor<const D: usize, C = Float> {
    raw: RawTensor,
    tape: Option<Tape>,
    _class: PhantomData<C>,
}

impl<const D: usize, C: DataClassTrait> Tensor<D, C> {
    pub(crate) fn from_raw(raw: RawTensor) -> Self {
        assert_eq!(raw.data_type().class(), C::data_class());
        assert_eq!(raw.shape().num_dims(), D);
        Self {
            raw,
            tape: None,
            _class: PhantomData,
        }
    }

    pub(crate) fn into_raw(self) -> RawTensor {
        self.raw
    }

    pub(crate) fn as_raw(&self) -> &RawTensor {
        &self.raw
    }

    pub(crate) fn as_raw_mut(&mut self) -> &mut RawTensor {
        &mut self.raw
    }

    pub fn is_virtual(&self) -> bool {
        self.raw.is_virtual()
    }

    pub fn async_start_eval(&self) {
        self.raw.async_start_eval();
    }

    pub fn device(&self) -> Device {
        self.raw.device()
    }

    pub fn shape(&self) -> [usize; D] {
        self.raw.shape().dims().try_into().expect("axis mismatch?")
    }

    pub fn data_type(&self) -> DataType {
        self.raw.data_type()
    }
}

/// Tensor shape operations.
impl<const D: usize, C: DataClassTrait> Tensor<D, C> {
    pub fn reshape<const D2: usize>(&self, new_shape: [usize; D2]) -> Tensor<D2, C> {
        let old_shape = self.shape();
        self.op(
            move |x| x.reshape(new_shape),
            move |_, flow| flow.reshape(old_shape),
        )
    }

    pub fn broadcast_to<const D2: usize>(&self, new_shape: [usize; D2]) -> Tensor<D2, C> {
        if &self.shape()[..] == &new_shape[..] {
            return Tensor {
                raw: self.raw.clone(),
                tape: self.tape.clone(),
                _class: PhantomData,
            };
        }

        let old_shape = self.shape();

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

        self.op(
            move |x| x.broadcast_to(new_shape),
            move |_, mut flow| {
                for (a, b) in apply_permutation_via_swaps(D2, &|x| axis_mapping[&x]) {
                    flow = flow.swap_dims(a, b);
                }

                let mut temp_shape = old_shape.to_vec();
                while temp_shape.len() < D2 {
                    temp_shape.push(1);
                }
                flow = flow
                    .reduce_sum(broadcast_axes.len() as u32)
                    .reshape(temp_shape);

                for (a, b) in apply_permutation_via_swaps(D2, &|x| inv_axis_mapping[&x]) {
                    flow = flow.swap_dims(a, b);
                }

                flow.reshape(old_shape)
            },
        )
    }

    pub fn swap_dims(&self, axis_a: usize, axis_b: usize) -> Self {
        self.op(
            move |x| x.swap_dims(axis_a, axis_b),
            move |_, flow| flow.swap_dims(axis_a, axis_b),
        )
    }

    pub fn transpose(&self) -> Self {
        self.swap_dims(D - 1, D - 2)
    }
}

/// Float-only operations.
impl<const D: usize> Tensor<D, Float> {
    pub fn from_slice(slice: impl AsDataSlice, shape: [usize; D], device: Device) -> Self {
        Self::from_raw(RawTensor::from_slice(slice, shape, device))
    }

    pub fn from_constant(constant: impl Into<Scalar>, shape: [usize; D], device: Device) -> Self {
        let constant = constant.into();
        assert_eq!(
            constant.data_type().class(),
            DataClass::Float,
            "called Tensor::from_constant with non-matching data type"
        );
        Self::from_raw(RawTensor::from_constant(constant, shape, device))
    }

    /// Creates a tensor of all zeros in the given shape.
    pub fn zeros(shape: [usize; D], data_type: DataType, device: Device) -> Self {
        let scalar = match data_type {
            DataType::F16 => Scalar::F16(f16::ZERO),
            DataType::Bf16 => Scalar::Bf16(bf16::ZERO),
            DataType::F32 => Scalar::F32(0.0),
            _ => panic!("illegal data type {data_type:?} for Tensor::zeros"),
        };
        Self::from_constant(scalar, shape, device)
    }

    /// Creates a tensor of all ones in the given shape.
    pub fn ones(shape: [usize; D], data_type: DataType, device: Device) -> Self {
        let scalar = match data_type {
            DataType::F16 => Scalar::F16(f16::ONE),
            DataType::Bf16 => Scalar::Bf16(bf16::ONE),
            DataType::F32 => Scalar::F32(1.0),
            _ => panic!("illegal data type {data_type:?} for Tensor::zeros"),
        };
        Self::from_constant(scalar, shape, device)
    }

    pub async fn to_vec_async<T: DataTypeConversion<Float>>(&self) -> Vec<T> {
        self.raw.clone().into_vec().await.to_floats()
    }

    pub fn to_vec<T: DataTypeConversion<Float>>(&self) -> Vec<T> {
        self.to_vec_async().block_on()
    }

    pub fn to_data_type(&self, new_dtype: DataType) -> Self {
        assert_eq!(
            new_dtype.class(),
            DataClass::Float,
            "cannot convert from float to non-float data class"
        );
        let old_dtype = self.data_type();
        self.op(
            move |x| x.into_data_type(new_dtype),
            move |_, flow| flow.into_data_type(old_dtype),
        )
    }

    pub fn enable_grad(&mut self) {
        if self.tape.is_none() {
            self.tape = Some(Tape::new_constant(self.raw.clone()));
        }
    }

    pub fn detach(&self) -> Self {
        Self {
            tape: None,
            raw: self.raw.clone(),
            _class: PhantomData,
        }
    }

    pub fn checkpoint(mut self) -> Self {
        self.tape = self.tape.map(|t| t.checkpoint(self.raw.clone()));
        self
    }

    /// # Panics
    /// Panics if gradient tracking is not enabled on this tensor.
    pub fn backward(&self) -> Gradients {
        let tape = self
            .tape
            .as_ref()
            .expect("called backward() on tensor that did not have gradient tracking enabled");
        tape.backward(Tensor::<1>::from_raw(RawTensor::from_constant(
            1.0f32,
            [1],
            self.raw.device(),
        )))
    }
}

impl Tensor<1, Float> {
    pub fn from_scalar<T>(scalar: T, device: Device) -> Self
    where
        for<'a> &'a [T]: AsDataSlice,
    {
        Self::from_slice(&[scalar][..], [1], device)
    }

    pub fn from_array<const N: usize, T>(array: [T; N], device: Device) -> Self
    where
        for<'a> &'a [T]: AsDataSlice,
    {
        Self::from_slice(&array[..], [N], device)
    }

    pub async fn to_scalar_async<T: DataTypeConversion<Float>>(&self) -> T {
        let vec = self.to_vec_async::<T>().await;
        assert_eq!(vec.len(), 1, "called to_scalar on a tensor of size != 1");
        vec[0]
    }

    pub fn to_scalar<T: DataTypeConversion<Float>>(&self) -> T {
        self.to_scalar_async().block_on()
    }
}

impl Tensor<2, Float> {
    pub fn from_array<const N1: usize, const N2: usize, T>(
        array: [[T; N2]; N1],
        device: Device,
    ) -> Self
    where
        for<'a> &'a [T]: AsDataSlice,
        T: Pod,
    {
        Self::from_slice(bytemuck::cast_slice(&array), [N1, N2], device)
    }
}

impl Tensor<3, Float> {
    pub fn from_array<const N1: usize, const N2: usize, const N3: usize, T>(
        array: [[[T; N3]; N2]; N1],
        device: Device,
    ) -> Self
    where
        for<'a> &'a [T]: AsDataSlice,
        T: Pod,
    {
        Self::from_slice(bytemuck::cast_slice(&array), [N1, N2, N3], device)
    }
}

impl Tensor<4, Float> {
    pub fn from_array<const N1: usize, const N2: usize, const N3: usize, const N4: usize, T>(
        array: [[[[T; N4]; N3]; N2]; N1],
        device: Device,
    ) -> Self
    where
        for<'a> &'a [T]: AsDataSlice,
        T: Pod,
    {
        Self::from_slice(bytemuck::cast_slice(&array), [N1, N2, N3, N4], device)
    }
}

/// Float math operations.
impl<const D: usize> Tensor<D, Float> {
    pub fn recip(&self) -> Self {
        self.op(|x| x.recip(), |x, flow| -(x.pow_scalar(-2.0)) * flow)
    }

    pub fn exp(&self) -> Self {
        self.op(|x| x.exp(), |x, flow| x.exp() * flow)
    }

    /// Natural logarithm.
    pub fn log(&self) -> Self {
        self.op(|x| x.log(), |x, flow| flow / x)
    }

    pub fn sigmoid(&self) -> Self {
        self.op(
            |x| x.sigmoid(),
            |x, flow| {
                let exp = (-x).exp();
                exp.clone() / (exp + 1.0).pow_scalar(2.0) * flow
            },
        )
    }

    pub fn pow(&self, power: impl AsTensor<D>) -> Self {
        self.op_binary(
            power.as_tensor(),
            |a, b| a.pow(b),
            |a, b, flow| {
                let dtype = flow.data_type();
                (b.clone() * a.pow(b - 1.0) * flow).into_data_type(dtype)
            },
            |a, b, flow| {
                let dtype = flow.data_type();
                (a.clone().pow(b) * a.log() * flow).into_data_type(dtype)
            },
        )
    }

    pub fn pow_scalar(&self, power: f32) -> Self {
        self.pow(Tensor::<1>::from_scalar(power, self.device()).broadcast_to(self.shape()))
    }

    pub fn sqrt(&self) -> Self {
        self.op(|x| x.sqrt(), |x, flow| flow / (x.sqrt() * 2.0))
    }

    pub fn sin(&self) -> Self {
        self.op(|x| x.sin(), |x, flow| flow * x.cos())
    }

    pub fn cos(&self) -> Self {
        self.op(|x| x.cos(), |x, flow| flow * -x.sin())
    }

    pub fn tan(&self) -> Self {
        self.op(|x| x.tan(), |x, flow| flow / (x.cos().pow_scalar(2.0)))
    }

    pub fn relu(&self) -> Self {
        self.op(
            |x| x.relu(),
            |x, flow| {
                let zero = RawTensor::from_constant(0.0f32, x.shape(), x.device());
                let one = RawTensor::from_constant(1.0f32, x.shape(), x.device());
                x.clone()
                    .compare_less_than(zero.clone())
                    .select(zero, one)
                    .into_data_type(flow.data_type())
                    * flow
            },
        )
    }

    pub fn max(&self, other: impl AsTensor<D>) -> Self {
        let compute_greater_mask =
            |a: &RawTensor, b: &RawTensor| a.clone().compare_greater_than(b.clone());
        let compute_equal_mask = |a: &RawTensor, b: &RawTensor| a.clone().compare_equal(b.clone());

        let compute_gradient = move |data: RawTensor, other: RawTensor| {
            let ones = RawTensor::from_constant(1.0f32, data.shape(), data.device());
            let zeros = RawTensor::from_constant(0.0f32, data.shape(), data.device());
            let halves = RawTensor::from_constant(0.5f32, data.shape(), data.device());

            let greater_mask = compute_greater_mask(&data, &other);
            let equal_mask = compute_equal_mask(&data, &other);

            greater_mask.select(ones, equal_mask.select(halves, zeros))
        };

        self.op_binary(
            other.as_tensor(),
            |a, b| a.max(b),
            move |a, b, flow| compute_gradient(a, b) * flow,
            move |a, b, flow| compute_gradient(b, a) * flow,
        )
    }

    pub fn min(&self, other: impl AsTensor<D>) -> Self {
        let compute_less_mask =
            |a: &RawTensor, b: &RawTensor| a.clone().compare_less_than(b.clone());
        let compute_equal_mask = |a: &RawTensor, b: &RawTensor| a.clone().compare_equal(b.clone());

        let compute_gradient = move |data: RawTensor, other: RawTensor| {
            let ones = RawTensor::from_constant(1.0f32, data.shape(), data.device());
            let zeros = RawTensor::from_constant(0.0f32, data.shape(), data.device());
            let halves = RawTensor::from_constant(0.5f32, data.shape(), data.device());

            let less_mask = compute_less_mask(&data, &other);
            let equal_mask = compute_equal_mask(&data, &other);

            less_mask.select(ones, equal_mask.select(halves, zeros))
        };

        self.op_binary(
            other.as_tensor(),
            |a, b| a.min(b),
            move |a, b, flow| compute_gradient(a, b) * flow,
            move |a, b, flow| compute_gradient(b, a) * flow,
        )
    }

    /// Row-major matrix multiplication: `self * rhs`.
    /// Optionally batched.
    ///
    /// The output type is implicitly converted to match the data type of `self`,
    /// regardless of the selected precision.
    pub fn matmul(&self, rhs: impl AsTensor<D>, precision: Precision) -> Self {
        self.op_binary(
            rhs.as_tensor(),
            move |a, b| a.matmul(b, precision),
            move |_a, b, flow| b.matmul(flow.transpose(), precision).transpose(),
            move |a, _b, flow| a.transpose().matmul(flow, precision),
        )
        .checkpoint()
    }

    /// 2D batched convolution of the image `self` with filter `filter`.
    ///
    /// Image layout is NHWC (channels last), as opposed to torch which
    /// uses NCHW. Filter layout is KRSC, where K is the number of output
    /// channels and R and S the filter dimensions.
    pub fn conv2d(&self, filter: impl AsTensor<4>, settings: Conv2dParams) -> Self {
        const {
            // TODO: potentially add support for greater number of batch dimensions?
            if D != 4 {
                panic!("2D convolution currently requires input dimension of 4");
            }
        }

        let input_size = [self.shape()[1], self.shape()[2]];

        self.op_binary(
            filter.as_tensor(),
            move |image, filter| image.conv2d(filter, settings),
            move |_, filter, flow| {
                // Compute image gradient
                flow.conv2d_backward_data(filter, settings, input_size)
            },
            move |image, _, flow| {
                // Compute filter gradient
                flow.conv2d_backward_filter(image, settings)
            },
        )
        .checkpoint()
    }

    pub fn reduce_sum<const D2: usize>(&self, depth: u32) -> Tensor<D2> {
        let shape = self.shape();
        let result_shape = self.raw.clone().reduce_sum(depth).shape();
        let dtype = self.data_type();
        self.op(
            move |x| x.reduce_sum(depth),
            move |_, flow| {
                let mut new_shape = [1usize; D];
                new_shape[..D2].copy_from_slice(result_shape.dims());
                flow.reshape(new_shape)
                    .broadcast_to(shape)
                    .into_data_type(dtype)
            },
        )
    }

    pub fn reduce_mean<const D2: usize>(&self, depth: u32) -> Tensor<D2> {
        let shape = self.shape();
        let stride = shape[shape.len() - depth as usize..]
            .iter()
            .copied()
            .product::<usize>();
        let result_shape = self.raw.clone().reduce_sum(depth).shape();
        let dtype = self.data_type();
        self.op(
            move |x| x.reduce_mean(depth),
            move |_, flow| {
                let mut new_shape = [1usize; D];
                new_shape[..D2].copy_from_slice(result_shape.dims());
                (flow.reshape(new_shape).broadcast_to(shape) * (stride as f32).recip())
                    .into_data_type(dtype)
            },
        )
    }

    pub fn reduce_min<const D2: usize>(&self, depth: u32) -> Tensor<D2> {
        let shape = self.shape();
        let dtype = self.data_type();
        self.op(
            move |x| x.reduce_min(depth),
            move |x, flow| {
                let mut temp_shape = [1usize; D];
                temp_shape[..D2].copy_from_slice(flow.shape().dims());

                let min = x
                    .clone()
                    .reduce_min(depth)
                    .reshape(temp_shape)
                    .broadcast_to(shape);

                let match_mask = min.compare_equal(x.clone());

                let num_matching = match_mask
                    .clone()
                    .into_data_type(DataType::F32)
                    .reduce_sum(depth)
                    .reshape(temp_shape)
                    .broadcast_to(shape);

                (match_mask.select(
                    num_matching.recip(),
                    RawTensor::from_constant(0.0f32, shape, x.device()),
                ) * flow.reshape(temp_shape).broadcast_to(shape))
                .into_data_type(dtype)
            },
        )
    }

    pub fn reduce_max<const D2: usize>(&self, depth: u32) -> Tensor<D2> {
        let shape = self.shape();
        let dtype = self.data_type();
        self.op(
            move |x| x.reduce_max(depth),
            move |x, flow| {
                let mut temp_shape = [1usize; D];
                temp_shape[..D2].copy_from_slice(flow.shape().dims());

                let min = x
                    .clone()
                    .reduce_max(depth)
                    .reshape(temp_shape)
                    .broadcast_to(shape);

                let match_mask = min.compare_equal(x.clone());
                let num_matching = match_mask
                    .clone()
                    .into_data_type(DataType::F16)
                    .reduce_sum(depth)
                    .reshape(temp_shape)
                    .broadcast_to(shape);

                (match_mask.select(
                    num_matching.recip(),
                    RawTensor::from_constant(0.0f32, shape, x.device()),
                ) * flow.reshape(temp_shape).broadcast_to(shape))
                .into_data_type(dtype)
            },
        )
    }
}

impl Tensor<2, Float> {
    pub fn log_softmax(&self) -> Tensor<2> {
        let max = self.reduce_max::<2>(1).broadcast_to(self.shape());
        self - &max
            - (self - max)
                .exp()
                .reduce_sum::<2>(1)
                .log()
                .broadcast_to(self.shape())
    }
}

impl<const D: usize, C: DataClassTrait> Debug for Tensor<D, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor").field("raw", &self.raw).finish()
    }
}

impl<const D: usize> Neg for &'_ Tensor<D> {
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        self.op(|x| -x, |_, flow| -flow)
    }
}

impl<const D: usize> Neg for Tensor<D> {
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<const D: usize, T: AsTensor<D>> Add<T> for &'_ Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = rhs.as_tensor();
        self.op_binary(rhs, |a, b| a + b, |_, _, flow| flow, |_, _, flow| flow)
    }
}

impl<const D: usize, T: AsTensor<D>> Add<T> for Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: T) -> Self::Output {
        &self + rhs
    }
}

impl<const D: usize> Add<f32> for &'_ Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: f32) -> Self::Output {
        self + Tensor::<1>::from_scalar(rhs, self.device()).broadcast_to(self.shape())
    }
}

impl<const D: usize> Add<f32> for Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: f32) -> Self::Output {
        &self + rhs
    }
}

impl<const D: usize, T: AsTensor<D>> Sub<T> for &'_ Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: T) -> Self::Output {
        self + -rhs.as_tensor()
    }
}

impl<const D: usize, T: AsTensor<D>> Sub<T> for Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: T) -> Self::Output {
        &self - rhs
    }
}

impl<const D: usize> Sub<f32> for &'_ Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: f32) -> Self::Output {
        self - Tensor::<1>::from_scalar(rhs, self.device()).broadcast_to(self.shape())
    }
}

impl<const D: usize> Sub<f32> for Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: f32) -> Self::Output {
        &self - rhs
    }
}

impl<const D: usize, T: AsTensor<D>> Mul<T> for &'_ Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: T) -> Self::Output {
        let dtype = self.data_type();
        self.op_binary(
            rhs.as_tensor(),
            |a, b| a * b,
            move |_, b, flow| (b * flow).into_data_type(dtype),
            move |a, _, flow| (a * flow).into_data_type(dtype),
        )
    }
}

impl<const D: usize, T: AsTensor<D>> Mul<T> for Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: T) -> Self::Output {
        &self * rhs
    }
}

impl<const D: usize> Mul<f32> for &'_ Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: f32) -> Self::Output {
        self * Tensor::<1>::from_scalar(rhs, self.device()).broadcast_to(self.shape())
    }
}

impl<const D: usize> Mul<f32> for Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: f32) -> Self::Output {
        &self * rhs
    }
}

impl<const D: usize, T: AsTensor<D>> Div<T> for &'_ Tensor<D> {
    type Output = Tensor<D>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: T) -> Self::Output {
        self * rhs.as_tensor().recip()
    }
}

impl<const D: usize, T: AsTensor<D>> Div<T> for Tensor<D> {
    type Output = Tensor<D>;

    fn div(self, rhs: T) -> Self::Output {
        &self / rhs
    }
}

impl<const D: usize> Div<f32> for &'_ Tensor<D> {
    type Output = Tensor<D>;

    fn div(self, rhs: f32) -> Self::Output {
        self / Tensor::<1>::from_scalar(rhs, self.device()).broadcast_to(self.shape())
    }
}

impl<const D: usize> Div<f32> for Tensor<D> {
    type Output = Tensor<D>;

    fn div(self, rhs: f32) -> Self::Output {
        &self / rhs
    }
}

/// Internal helpers for autodiff.
impl<const D: usize, C: DataClassTrait> Tensor<D, C> {
    fn op<const D2: usize>(
        &self,
        compute: impl Fn(RawTensor) -> RawTensor + Send + Sync + 'static,
        compute_gradient: impl Fn(RawTensor, RawTensor) -> RawTensor + Send + Sync + 'static,
    ) -> Tensor<D2, C> {
        let result = compute(self.raw.clone());
        let tape = self
            .tape
            .as_ref()
            .map(|tape| tape.clone().append_unary(compute, compute_gradient));
        Tensor {
            raw: result,
            tape,
            _class: PhantomData,
        }
    }

    fn op_binary<const D2: usize, const D3: usize>(
        &self,
        rhs: &Tensor<D2, C>,
        compute: impl Fn(RawTensor, RawTensor) -> RawTensor + Send + Sync + 'static,
        compute_gradient_left: impl Fn(RawTensor, RawTensor, RawTensor) -> RawTensor
        + Send
        + Sync
        + 'static,
        compute_gradient_right: impl Fn(RawTensor, RawTensor, RawTensor) -> RawTensor
        + Send
        + Sync
        + 'static,
    ) -> Tensor<D3, C> {
        let result = compute(self.raw.clone(), rhs.raw.clone());

        let tape = if self.tape.is_some() || rhs.tape.is_some() {
            let lhs_tape = self
                .tape
                .clone()
                .unwrap_or_else(|| Tape::new_constant(self.raw.clone()));
            let rhs_tape = rhs
                .tape
                .clone()
                .unwrap_or_else(|| Tape::new_constant(rhs.raw.clone()));
            Some(lhs_tape.append_binary(
                rhs_tape,
                compute,
                compute_gradient_left,
                compute_gradient_right,
            ))
        } else {
            None
        };

        Tensor {
            raw: result,
            tape,
            _class: PhantomData,
        }
    }
}

pub trait AsTensor<const D: usize, C: DataClassTrait = Float> {
    fn as_tensor(&self) -> &Tensor<D, C>;
}

impl<const D: usize, C: DataClassTrait> AsTensor<D, C> for Tensor<D, C> {
    fn as_tensor(&self) -> &Tensor<D, C> {
        self
    }
}

impl<const D: usize, C: DataClassTrait> AsTensor<D, C> for &'_ Tensor<D, C> {
    fn as_tensor(&self) -> &Tensor<D, C> {
        self
    }
}

impl<const D: usize> AsTensor<D, Float> for Param<D> {
    fn as_tensor(&self) -> &Tensor<D, Float> {
        self.value()
    }
}

impl<const D: usize> AsTensor<D, Float> for &'_ Param<D> {
    fn as_tensor(&self) -> &Tensor<D, Float> {
        self.value()
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
    use rand::{SeedableRng, seq::SliceRandom};
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
