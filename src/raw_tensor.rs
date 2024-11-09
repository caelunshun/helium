use crate::{
    backend::{Backend, BackendExt},
    cache::Cache,
    conv::Conv2dParams,
    data_type::{AsDataSlice, DataClass, DataSlice, DataType, DataVec, Scalar},
    device::Device,
    opgraph::{
        op,
        op::{
            BinaryPointwise, BinaryPointwiseOp, BroadcastAxis, CompareOp, Op, Reduce, ReduceOp,
            UnaryPointwise, UnaryPointwiseOp,
        },
        Descriptor, NodeId, OpGraph,
    },
    shape::Shape,
};
use ahash::AHashMap;
use parking_lot::Mutex;
use slotmap::{Key, SecondaryMap};
use std::{
    fmt::Debug,
    future::Future,
    ops::{Add, Div, Mul, Neg, Sub},
    pin::Pin,
    sync::{Arc, Weak},
    task::{Context, Poll, Waker},
};

/// Raw type-erased tensor, with dynamic dimension,
/// data class, and data type. Does not store backpropagation
/// information.
#[derive(Clone)]
pub struct RawTensor {
    inner: Arc<Mutex<TensorInner>>,
    device: Device,
}

impl Debug for RawTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner.lock().data {
            Data::Virtual(data) => f
                .debug_struct("VirtualTensor")
                .field("node", &data.node)
                .field("graph", &Arc::as_ptr(&data.graph))
                .finish(),
            Data::Concrete(_) => f.debug_struct("ConcreteTensor").finish(),
        }
    }
}

impl RawTensor {
    pub fn is_virtual(&self) -> bool {
        matches!(&self.inner.lock().data, Data::Virtual(_))
    }

    pub fn async_start_eval(&self) {
        self.make_concrete();
    }

    pub fn num_dims(&self) -> usize {
        self.shape().num_dims()
    }

    pub fn from_slice(slice: impl AsDataSlice, shape: impl Into<Shape>, device: Device) -> Self {
        let slice = slice.as_data_slice();
        let shape = shape.into();

        if slice.len() == 1 {
            static SCALAR_CACHE: Cache<(Device, Scalar), RawTensor> = Cache::with_capacity(16384);
            SCALAR_CACHE.get_or_insert(&(device, slice.get(0)), move || {
                Self::from_slice_impl(slice, shape, device)
            })
        } else {
            Self::from_slice_impl(slice, shape, device)
        }
    }

    fn from_slice_impl(slice: DataSlice, shape: Shape, device: Device) -> Self {
        if slice.data_type() == DataType::Bool {
            assert_eq!(
                shape.num_elements(),
                (slice.len() + 31) / 32 * 32,
                "product of tensor dimensions must equal the length of the data"
            );
        } else {
            assert_eq!(
                shape.num_elements(),
                slice.len(),
                "product of tensor dimensions must equal the length of the data"
            );
        }

        let inner = Arc::new(Mutex::new(TensorInner {
            data: Data::Concrete(match device {
                #[cfg(feature = "cuda")]
                Device::Cuda(device) => ConcreteData::Cuda(
                    shape,
                    crate::cuda::Cuda.create_tensor_with_data(slice, device),
                ),
                #[cfg(feature = "cpu")]
                Device::Cpu => todo!(),
            }),
        }));

        Self { inner, device }
    }

    pub fn from_constant(
        value: impl Into<Scalar>,
        shape: impl Into<Shape>,
        device: Device,
    ) -> Self {
        let shape = shape.into();
        assert!(shape.num_elements() > 0);
        let builder = OpGraphBuilderHandle::new(Mutex::new(OpGraphBuilder::new(device)));
        Self::from_op(
            &builder,
            Op::Constant(op::Constant {
                value: value.into(),
                shape: shape,
            }),
        )
    }

    pub fn into_vec(self) -> IntoVec {
        self.make_concrete();
        let inner = self.inner.lock();
        let Data::Concrete(data) = &inner.data else {
            unreachable!("make_concrete() was called")
        };

        let vec = Arc::new(Mutex::new(None));
        let waker = Arc::new(Mutex::new(None));
        let fut = IntoVec {
            vec: vec.clone(),
            waker: waker.clone(),
        };

        match data {
            #[cfg(feature = "cuda")]
            ConcreteData::Cuda(_, tensor) => {
                let Device::Cuda(device) = self.device else {
                    unreachable!("mismatched device")
                };
                crate::cuda::Cuda.download_tensor(
                    tensor,
                    move |result| {
                        *vec.lock() = Some(result);
                        if let Some(waker) = waker.lock().take() {
                            waker.wake();
                        }
                    },
                    device,
                );
            }
            #[cfg(feature = "cpu")]
            ConcreteData::Cpu => todo!(),
        }

        fut
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn shape(&self) -> Shape {
        self.inner.lock().data.shape()
    }

    pub fn data_type(&self) -> DataType {
        self.inner.lock().data_type()
    }

    pub fn recip(self) -> Self {
        self.op_unary_pointwise(UnaryPointwiseOp::Recip)
    }

    /// Row-major matrix multiplication.
    pub fn matmul(self, rhs: Self) -> Self {
        let d = self.num_dims();
        if d < 2 {
            panic!("matrix multiplication requires tensors of dimension >= 2");
        }

        let shape_lhs = self.shape();
        let shape_rhs = rhs.shape();

        assert_eq!(
            shape_lhs[d - 1],
            shape_rhs[d - 2],
            "invalid dimensions for matmul: {}x{} (lhs) is not compatible with {}x{} (rhs)",
            shape_lhs[d - 2],
            shape_lhs[d - 1],
            shape_rhs[d - 2],
            shape_rhs[d - 1],
        );

        assert_eq!(
            self.data_type(),
            rhs.data_type(),
            "matmul only supported when A and B have the same data type"
        );

        if d > 2 {
            assert_eq!(
                &self.shape().dims()[..d - 2],
                &rhs.shape().dims()[..d - 2],
                "for batched matmul, all batch dimensions must have the same size"
            );
        }

        let (cx, _) = self.make_graph();
        let cx = rhs.to_graph(cx);
        Self::from_op(
            &cx,
            Op::Matmul(op::Matmul {
                input_a: self.node(&cx),
                input_b: rhs.node(&cx),
            }),
        )
    }

    /// Batched 2D convolution. Image tensor layout is `NHWC` (channel dimension last).
    /// Kernel tensor layout is `KRSC`.
    /// `N` = batch size
    /// `H` = height
    /// `W` = width
    /// `C` = input channel count
    /// `K` = output channel count
    /// `R` = kernel height
    /// `S` = kernel width
    pub fn conv2d(self, filter: Self, settings: Conv2dParams) -> Self {
        settings.validate();

        let image_shape = self.shape();
        let filter_shape = filter.shape();

        assert_eq!(
            image_shape.num_dims(),
            4,
            "conv2d requires exactly 4 image dimensions (including a batch dimension)"
        );
        assert_eq!(
            filter_shape.num_dims(),
            4,
            "conv2d requires exactly 4 filter dimensions: KRSC"
        );

        assert_eq!(
            image_shape.dim_at(3),
            settings.in_channels,
            "dimension 3 must equal the number of channels"
        );

        assert_eq!(filter_shape.dim_at(1), settings.kernel_size[0]);
        assert_eq!(filter_shape.dim_at(2), settings.kernel_size[1]);
        assert_eq!(filter_shape.dim_at(0), settings.out_channels);
        assert_eq!(filter_shape.dim_at(3), settings.in_channels);

        assert_eq!(
            filter.data_type(),
            self.data_type(),
            "conv2d: image and filter must have same data type"
        );
        assert_eq!(
            self.data_type().class(),
            DataClass::Float,
            "conv2d only supports float data types"
        );

        let (cx, _) = self.make_graph();
        let cx = filter.to_graph(cx);
        Self::from_op(
            &cx,
            Op::Conv(op::Conv {
                settings,
                image: self.node(&cx),
                filter: filter.node(&cx),
            }),
        )
    }

    pub fn conv2d_backward_data(
        self,
        filter: Self,
        settings: Conv2dParams,
        input_size: [usize; 2],
    ) -> Self {
        // TODO: validation
        // (this function is only used internally by Tensor autodiff)
        let (cx, _) = self.make_graph();
        let cx = filter.to_graph(cx);
        Self::from_op(
            &cx,
            Op::ConvBackwardData(op::ConvBackwardData {
                settings,
                flow: self.node(&cx),
                filter: filter.node(&cx),
                input_size,
            }),
        )
    }

    pub fn conv2d_backward_filter(self, image: Self, settings: Conv2dParams) -> Self {
        // TODO: validation
        // (this function is only used internally by Tensor autodiff)
        let (cx, _) = self.make_graph();
        let cx = image.to_graph(cx);
        Self::from_op(
            &cx,
            Op::ConvBackwardFilter(op::ConvBackwardFilter {
                settings,
                flow: self.node(&cx),
                image: image.node(&cx),
            }),
        )
    }

    /// Swaps last two dimensions.
    pub fn transpose(self) -> Self {
        let num_dims = self.num_dims();
        if num_dims < 2 {
            panic!("transpose requires at least two dimensions");
        }

        self.swap_dims(num_dims - 1, num_dims - 2)
    }

    /// Swaps the given two dimensions.
    pub fn swap_dims(self, axis_a: usize, axis_b: usize) -> Self {
        assert!(
            axis_a < self.num_dims(),
            "swap_dims: axis {axis_a} out of bounds for tensor of dimension {}",
            self.num_dims()
        );
        assert!(
            axis_b < self.num_dims(),
            "swap_dims: axis {axis_b} out of bounds for tensor of dimension {}",
            self.num_dims()
        );

        if axis_a == axis_b {
            // No-op.
            return self;
        }

        let (cx, this) = self.make_graph();
        Self::from_op(
            &cx,
            Op::SwapDims(op::SwapDims {
                input: this,
                axis_a,
                axis_b,
            }),
        )
    }

    /// Broadcasts the tensor to the given shape.
    ///
    /// Dimensions of the output must match dimensions of `self`
    /// except where the `self` dimension is of size 1.
    pub fn broadcast_to(self, new_shape: impl Into<Shape>) -> RawTensor {
        let new_shape = new_shape.into();
        let d = self.num_dims();
        let d2 = new_shape.num_dims();

        if new_shape == self.shape() {
            // No need for broadcast, optimize out by returning
            // `self`.
            return self;
        }

        assert!(
            new_shape.dims().iter().all(|&x| x != 0),
            "dimension cannot be of size zero"
        );

        let mut broadcast_axes = Vec::new();

        for (reverse_index, (old, new)) in self
            .shape()
            .dims()
            .iter()
            .copied()
            .rev()
            .zip(new_shape.dims().iter().rev())
            .enumerate()
        {
            let index = d2 - reverse_index - 1;
            if old != 1 {
                assert_eq!(
                    old, *new,
                    "cannot broadcast axis {index} that is not of size 1 (actual size is {old})"
                );
            }

            if old == 1 && *new != 1 {
                broadcast_axes.push(BroadcastAxis::new(index, *new));
            }
        }
        if d2 > d {
            for i in 0..(d2 - d) {
                broadcast_axes.push(BroadcastAxis::new(i, new_shape[i]));
            }
        }

        let (cx, this) = self.make_graph();
        RawTensor::from_op(
            &cx,
            Op::Broadcast(op::Broadcast {
                input: this,
                new_dim_count: d2,
                broadcast_axes,
            }),
        )
    }

    pub fn pow(self, other: Self) -> Self {
        self.broadcast_to(other.shape())
            .op_binary_pointwise(&other, BinaryPointwiseOp::Pow)
    }

    pub fn pow_scalar(self, power: f32) -> Self {
        let rhs = RawTensor::from_slice(&[power][..], [1], self.device).broadcast_to(self.shape());
        self.pow(rhs)
    }

    /// Natural logarithm.
    pub fn log(self) -> Self {
        self.op_unary_pointwise(UnaryPointwiseOp::Log)
    }

    /// `e^self`.`
    pub fn exp(self) -> Self {
        self.op_unary_pointwise(UnaryPointwiseOp::Exp)
    }

    pub fn sin(self) -> Self {
        self.op_unary_pointwise(UnaryPointwiseOp::Sin)
    }

    pub fn cos(self) -> Self {
        self.op_unary_pointwise(UnaryPointwiseOp::Cos)
    }

    pub fn tan(self) -> Self {
        self.op_unary_pointwise(UnaryPointwiseOp::Tan)
    }

    pub fn sqrt(self) -> Self {
        self.op_unary_pointwise(UnaryPointwiseOp::Sqrt)
    }

    pub fn sigmoid(self) -> Self {
        self.op_unary_pointwise(UnaryPointwiseOp::Sigmoid)
    }

    #[expect(unused)]
    pub fn min(self, other: Self) -> Self {
        self.op_binary_pointwise(&other, BinaryPointwiseOp::Min)
    }

    #[expect(unused)]
    pub fn max(self, other: Self) -> Self {
        self.op_binary_pointwise(&other, BinaryPointwiseOp::Max)
    }

    pub fn relu(self) -> Self {
        self.op_unary_pointwise(UnaryPointwiseOp::Relu)
    }

    pub fn into_data_type(self, dtype: DataType) -> Self {
        if dtype == self.data_type() {
            return self;
        }

        let (cx, this) = self.make_graph();
        Self::from_op(
            &cx,
            Op::ChangeDataType(op::ChangeDataType {
                input: this,
                target_type: dtype,
            }),
        )
    }

    fn compare_op(self, rhs: Self, op: CompareOp) -> Self {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "mismatched shape for compare op {op:?}"
        );
        assert_eq!(
            self.data_type().class(),
            rhs.data_type().class(),
            "mismatched data class for compare op {op:?}"
        );
        let (cx, _) = self.make_graph();
        let cx = rhs.to_graph(cx);
        Self::from_op(
            &cx,
            Op::Compare(op::Compare {
                op,
                lhs: self.node(&cx),
                rhs: rhs.node(&cx),
            }),
        )
    }

    pub fn compare_equal(self, rhs: Self) -> Self {
        self.compare_op(rhs, CompareOp::Equal)
    }

    #[expect(unused)]
    pub fn compare_not_equal(self, rhs: Self) -> Self {
        self.compare_op(rhs, CompareOp::NotEqual)
    }

    pub fn compare_less_than(self, rhs: Self) -> Self {
        self.compare_op(rhs, CompareOp::LessThan)
    }

    #[expect(unused)]
    pub fn compare_less_than_or_equal(self, rhs: Self) -> Self {
        self.compare_op(rhs, CompareOp::LessThanOrEqual)
    }

    #[expect(unused)]
    pub fn compare_greater_than(self, rhs: Self) -> Self {
        self.compare_op(rhs, CompareOp::GreaterThan)
    }

    #[expect(unused)]
    pub fn compare_greater_than_or_equal(self, rhs: Self) -> Self {
        self.compare_op(rhs, CompareOp::GreaterThanOrEqual)
    }

    pub fn select(self, true_case: Self, false_case: Self) -> Self {
        assert_eq!(self.data_type(), DataType::Bool);
        assert_eq!(
            true_case.shape(),
            false_case.shape(),
            "shape mismatch for select"
        );
        assert_eq!(self.shape(), true_case.shape(), "shape mismatch for select");
        assert_eq!(
            true_case.data_type().class(),
            false_case.data_type().class(),
            "data class mismatch for select"
        );

        let (cx, _) = self.make_graph();
        let cx = true_case.to_graph(cx);
        let cx = false_case.to_graph(cx);

        Self::from_op(
            &cx,
            Op::Select(op::Select {
                lhs: false_case.node(&cx),
                rhs: true_case.node(&cx),
                selector: self.node(&cx),
            }),
        )
    }

    /// Performs sum reduction along the last `depth` dimensions
    /// of the tensor. The last `depth` dimensions are replaced
    /// with a single dimension of length 1.
    pub fn reduce_sum(self, depth: u32) -> RawTensor {
        self.op_reduce(ReduceOp::Sum, depth)
    }

    /// Performs mean reduction along the last `depth` dimensions
    /// of the tensor. The last `depth` dimensions are replaced
    /// with a single dimension of length 1.
    pub fn reduce_mean(self, depth: u32) -> RawTensor {
        self.op_reduce(ReduceOp::Mean, depth)
    }

    /// Performs min reduction along the last `depth` dimensions
    /// of the tensor. The last `depth` dimensions are replaced
    /// with a single dimension of length 1.
    pub fn reduce_min(self, depth: u32) -> RawTensor {
        self.op_reduce(ReduceOp::Min, depth)
    }

    /// Performs max reduction along the last `depth` dimensions
    /// of the tensor. The last `depth` dimensions are replaced
    /// with a single dimension of length 1.
    pub fn reduce_max(self, depth: u32) -> RawTensor {
        self.op_reduce(ReduceOp::Max, depth)
    }

    pub fn reshape(self, new_shape: impl Into<Shape>) -> RawTensor {
        let new_shape = new_shape.into();

        if new_shape == self.shape() {
            // No-op
            return self;
        }

        assert_eq!(
            self.shape().num_elements(),
            new_shape.num_elements(),
            "reshape() called with non-matching shape"
        );

        let (cx, this) = self.make_graph();
        RawTensor::from_op(
            &cx,
            Op::Reshape(op::Reshape {
                input: this,
                new_shape,
            }),
        )
    }
}

impl Neg for RawTensor {
    type Output = RawTensor;

    fn neg(self) -> Self::Output {
        self.op_unary_pointwise(UnaryPointwiseOp::Neg)
    }
}

impl Add for RawTensor {
    type Output = RawTensor;

    fn add(self, rhs: Self) -> Self::Output {
        self.op_binary_pointwise(&rhs, BinaryPointwiseOp::Add)
    }
}

impl Add<f32> for RawTensor {
    type Output = RawTensor;

    fn add(self, rhs: f32) -> Self::Output {
        let rhs = RawTensor::from_slice(&[rhs][..], [1], self.device).broadcast_to(self.shape());
        self + rhs
    }
}

impl Sub for RawTensor {
    type Output = RawTensor;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl Sub<f32> for RawTensor {
    type Output = RawTensor;

    fn sub(self, rhs: f32) -> Self::Output {
        self + -rhs
    }
}

impl Mul for RawTensor {
    type Output = RawTensor;

    fn mul(self, rhs: Self) -> Self::Output {
        self.op_binary_pointwise(&rhs, BinaryPointwiseOp::Mul)
    }
}
impl Mul<f32> for RawTensor {
    type Output = RawTensor;

    fn mul(self, rhs: f32) -> Self::Output {
        let rhs = RawTensor::from_slice(&[rhs][..], [1], self.device).broadcast_to(self.shape());
        self * rhs
    }
}

impl Div for RawTensor {
    type Output = RawTensor;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.recip()
    }
}

impl Div<f32> for RawTensor {
    type Output = RawTensor;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: f32) -> Self::Output {
        self * rhs.recip()
    }
}

/// Internal functions.
impl RawTensor {
    fn from_op(cx_arc: &OpGraphBuilderHandle, op: Op) -> Self {
        let mut cx = cx_arc.lock();
        let device = cx.device;
        let inner = Arc::new(Mutex::new(TensorInner {
            data: Data::Virtual(VirtualData {
                graph: Arc::clone(cx_arc),
                node: NodeId::null(),
                data_type: op
                    .output_descriptor(|node| cx.op_graph.get(node).descriptor().clone())
                    .data_type,
            }),
        }));

        let node_id = cx.new_op(op, &inner);

        let mut guard = inner.lock();
        let Data::Virtual(data) = &mut guard.data else {
            unreachable!()
        };
        data.node = node_id;
        drop(guard);

        Self { inner, device }
    }

    /// If the tensor is not already attached to a graph,
    /// we create a new graph. Then, we return the graph
    /// and this tensor's node ID in the graph.
    fn make_graph(&self) -> (OpGraphBuilderHandle, NodeId) {
        let guard = self.inner.lock();
        match &guard.data {
            Data::Virtual(virt) => (virt.graph.clone(), virt.node),
            Data::Concrete(_) => {
                let cx = Arc::new(Mutex::new(OpGraphBuilder::new(self.device)));
                let node = cx
                    .lock()
                    .new_input(guard.shape(), guard.data_type(), &self.inner);
                (cx, node)
            }
        }
    }

    fn node(&self, cx: &OpGraphBuilderHandle) -> NodeId {
        match &self.inner.lock().data {
            Data::Virtual(virt) => {
                debug_assert!(Arc::ptr_eq(cx, &virt.graph));
                virt.node
            }
            Data::Concrete(_) => cx.lock().inputs_memoized[&(Arc::as_ptr(&self.inner) as usize)],
        }
    }

    /// Moves the tensor onto the given graph. If the tensor
    /// is concrete, this registers the tensor as an input
    /// to the graph. If the tensor is virtual, then we merge
    /// the two graphs if they differ.
    fn to_graph(&self, builder: OpGraphBuilderHandle) -> OpGraphBuilderHandle {
        let guard = self.inner.lock();
        match &guard.data {
            Data::Virtual(virt) => {
                if Arc::ptr_eq(&virt.graph, &builder) {
                    builder
                } else {
                    let cx2 = virt.graph.clone();
                    drop(guard);
                    OpGraphBuilder::merge(cx2, builder)
                }
            }
            Data::Concrete(_) => {
                builder
                    .lock()
                    .new_input(guard.shape(), guard.data_type(), &self.inner);
                builder
            }
        }
    }

    fn op_binary_pointwise(&self, rhs: &Self, op: BinaryPointwiseOp) -> Self {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "shape mismatch for binary op {op:?}: lhs has shape {:?} while rhs has shape {:?}",
            self.shape(),
            rhs.shape()
        );
        let (cx, _) = self.make_graph();
        let cx = rhs.to_graph(cx);
        RawTensor::from_op(
            &cx,
            Op::BinaryPointwise(BinaryPointwise {
                lhs: self.node(&cx),
                rhs: rhs.node(&cx),
                op,
            }),
        )
    }

    fn op_unary_pointwise(&self, op: UnaryPointwiseOp) -> Self {
        let (cx, this) = self.make_graph();
        RawTensor::from_op(&cx, Op::UnaryPointwise(UnaryPointwise { input: this, op }))
    }

    fn op_reduce(self, op: ReduceOp, depth: u32) -> RawTensor {
        let input_dtype = self.data_type();
        assert_eq!(
            input_dtype.class(),
            DataClass::Float,
            "reductions only supported for float types"
        );

        if depth == 0 {
            return self;
        }

        assert!(
            depth <= self.num_dims() as u32,
            "reduction depth cannot exceed number of dimensions"
        );

        let (cx, this) = self.make_graph();
        let result = RawTensor::from_op(
            &cx,
            Op::Reduce(Reduce {
                input: this,
                op,
                depth,
            }),
        );

        // Internally, reductions always output in `f32` precision.
        // Convert back to the input precision.
        result.into_data_type(input_dtype)
    }

    fn make_concrete(&self) {
        if let Data::Concrete(_) = &self.inner.lock().data {
            return;
        }

        self.make_graph().0.lock().resolve();
    }
}

struct TensorInner {
    data: Data,
}

impl TensorInner {
    pub fn shape(&self) -> Shape {
        match &self.data {
            Data::Virtual(virt) => virt.shape(),
            Data::Concrete(conc) => conc.shape().clone(),
        }
    }

    pub fn data_type(&self) -> DataType {
        match &self.data {
            Data::Virtual(virt) => virt.data_type,
            Data::Concrete(conc) => match conc {
                #[cfg(feature = "cuda")]
                ConcreteData::Cuda(_, tensor) => tensor.data_type(),
                #[cfg(feature = "cpu")]
                ConcreteData::Cpu => todo!(),
            },
        }
    }
}

enum Data {
    Virtual(VirtualData),
    Concrete(ConcreteData),
}

impl Data {
    fn shape(&self) -> Shape {
        match self {
            Data::Virtual(data) => data.shape(),
            Data::Concrete(data) => data.shape().clone(),
        }
    }
}

struct VirtualData {
    graph: OpGraphBuilderHandle,
    node: NodeId,
    data_type: DataType,
}

impl VirtualData {
    pub fn shape(&self) -> Shape {
        self.graph
            .lock()
            .op_graph
            .get(self.node)
            .descriptor()
            .shape
            .clone()
    }
}

enum ConcreteData {
    #[cfg(feature = "cuda")]
    Cuda(Shape, <crate::cuda::Cuda as Backend>::TensorStorage),
    #[cfg(feature = "cpu")]
    #[expect(unused)]
    Cpu,
}

impl ConcreteData {
    fn shape(&self) -> &Shape {
        match self {
            ConcreteData::Cuda(shape, _) => shape,
            ConcreteData::Cpu => todo!(),
        }
    }
}

type OpGraphBuilderHandle = Arc<Mutex<OpGraphBuilder>>;

struct OpGraphBuilder {
    device: Device,
    op_graph: OpGraph,
    inputs: SecondaryMap<NodeId, Arc<Mutex<TensorInner>>>,
    /// Maps address of TensorInner to the node representing
    /// it, for input (concrete) tensors.
    inputs_memoized: AHashMap<usize, NodeId>,
    node_to_tensor: SecondaryMap<NodeId, Weak<Mutex<TensorInner>>>,
}

impl OpGraphBuilder {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            op_graph: OpGraph::new(),
            inputs: SecondaryMap::default(),
            inputs_memoized: AHashMap::default(),
            node_to_tensor: SecondaryMap::default(),
        }
    }

    pub fn merge(
        first_handle: OpGraphBuilderHandle,
        second_handle: OpGraphBuilderHandle,
    ) -> OpGraphBuilderHandle {
        let mut first = first_handle.lock();
        let mut second = second_handle.lock();

        if first.op_graph.num_nodes() > second.op_graph.num_nodes() {
            drop(first);
            drop(second);
            return Self::merge(second_handle, first_handle);
        }

        let node_mapping = first.op_graph.merge_into(&mut second.op_graph);

        for (old_node_id, tensor) in &first.node_to_tensor {
            let new_node_id = node_mapping[old_node_id];
            if let Some(tensor) = tensor.upgrade() {
                let mut guard = tensor.lock();
                if let Data::Virtual(virt) = &mut guard.data {
                    virt.graph = Arc::clone(&second_handle);
                    virt.node = new_node_id;
                }
            }
            second.node_to_tensor.insert(new_node_id, tensor.clone());
        }

        for (old_input_id, tensor) in first.inputs.drain() {
            let new_input_id = node_mapping[old_input_id];
            second.inputs.insert(new_input_id, tensor);
        }

        for (addr, node) in first.inputs_memoized.drain() {
            second.inputs_memoized.insert(addr, node_mapping[node]);
        }

        first.node_to_tensor.clear();

        drop(second);
        second_handle
    }

    pub fn new_input(
        &mut self,
        shape: Shape,
        data_type: DataType,
        tensor: &Arc<Mutex<TensorInner>>,
    ) -> NodeId {
        *self
            .inputs_memoized
            .entry(Arc::as_ptr(tensor) as usize)
            .or_insert_with(|| {
                let id = self.op_graph.new_input(Descriptor { shape, data_type });
                self.inputs.insert(id, Arc::clone(tensor));
                self.node_to_tensor.insert(id, Arc::downgrade(tensor));
                id
            })
    }

    pub fn new_op(&mut self, op: Op, owner: &Arc<Mutex<TensorInner>>) -> NodeId {
        let node = self.op_graph.new_op(op);
        self.node_to_tensor.insert(node, Arc::downgrade(owner));
        node
    }

    pub fn resolve(&mut self) {
        debug_assert!(self.op_graph.outputs().is_empty());
        // Mark tensors that are still alive as outputs.
        // This ensures these tensors are not virtualized by the backend,
        // as their data may be used later.
        for (node, tensor) in &self.node_to_tensor {
            if tensor.strong_count() > 0 {
                self.op_graph.new_output(node);
            }
        }

        match &self.device {
            #[cfg(feature = "cuda")]
            Device::Cuda(device_index) => {
                let mut inputs = SecondaryMap::default();
                for &node_id in self.op_graph.inputs() {
                    let tensor = self.node_to_tensor[node_id]
                        .upgrade()
                        .expect("tensor is referenced as input to graph, but no longer alive?");
                    let storage = match &tensor.lock().data {
                        Data::Concrete(ConcreteData::Cuda(_, storage)) => storage.clone(),
                        Data::Concrete(_) => panic!("tensor must be on CUDA device"),
                        Data::Virtual(_) => unreachable!("input tensor must be concrete"),
                    };
                    inputs.insert(node_id, storage);
                }

                let storages =
                    crate::cuda::Cuda.execute_graph(*device_index, self.op_graph.clone(), inputs);
                for (node_id, storage) in storages {
                    if let Some(tensor) = self.node_to_tensor.get(node_id).and_then(|t| t.upgrade())
                    {
                        tensor.lock().data = Data::Concrete(ConcreteData::Cuda(
                            self.op_graph.get(node_id).descriptor().shape.clone(),
                            storage,
                        ));
                    } else {
                        panic!();
                    }
                }
            }
            #[cfg(feature = "cpu")]
            Device::Cpu => todo!(),
        }

        self.inputs.clear();
    }
}

/// `Future` returned from `RawTensor::into_vec`.
pub struct IntoVec {
    vec: Arc<Mutex<Option<DataVec>>>,
    waker: Arc<Mutex<Option<Waker>>>,
}

impl Future for IntoVec {
    type Output = DataVec;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        *self.waker.lock() = Some(cx.waker().clone());
        match self.vec.lock().take() {
            Some(vec) => Poll::Ready(vec),
            None => Poll::Pending,
        }
    }
}
