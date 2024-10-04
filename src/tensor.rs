use crate::{
    data_type::{DataType, DataTypeConversion},
    device::Device,
    opgraph::{
        op,
        op::{
            BinaryPointwise, BinaryPointwiseOp, BroadcastAxis, Op, Reduce, ReduceOp, RestrctureOp,
            UnaryPointwise, UnaryPointwiseOp,
        },
        Descriptor, NodeId, OpGraph, Var, VarId, VarMap,
    },
    shape::Shape,
};
use parking_lot::Mutex;
use slotmap::{Key, SecondaryMap};
use std::{
    ops::{Add, Div, Mul, Neg, Sub},
    sync::{Arc, Weak},
};

/// Represents a tensor of dimension `D`.
///
/// Tensor data is immutable. Tensors can be cheaply
/// cloned like an `Arc`.
#[derive(Clone)]
pub struct Tensor<const D: usize> {
    inner: Arc<Mutex<TensorInner>>,
    device: Device,
}

/// API functions.
impl<const D: usize> Tensor<D> {
    pub fn from_vec<T: DataTypeConversion>(vec: Vec<T>, shape: [usize; D], device: Device) -> Self {
        assert_eq!(
            shape.iter().copied().product::<usize>(),
            vec.len(),
            "product of tensor dimensions must equal the length of the data"
        );

        let builder = Arc::new(Mutex::new(OpGraphBuilder::new(device)));
        let inner = Arc::new(Mutex::new(TensorInner {
            data: Data::Virtual(VirtualData {
                graph: builder.clone(),
                node: NodeId::null(),
                data_type: T::data_type(),
            }),
        }));

        let mut builder = builder.lock();

        let data_var = VarId::new();
        builder.vars.insert(
            data_var,
            Var::Tensor(T::into_data_vec(vec), Shape::new(shape)),
        );
        let node = builder.new_op(
            Op::UploadTensor(op::UploadTensor {
                data_var,
                descriptor: Descriptor {
                    shape: Shape::new(shape),
                    data_type: T::data_type(),
                },
            }),
            &inner,
        );

        match &mut inner.lock().data {
            Data::Virtual(data) => data.node = node,
            _ => unreachable!(),
        }

        Self { inner, device }
    }

    pub fn from_slice<T: DataTypeConversion>(
        slice: &[T],
        shape: [usize; D],
        device: Device,
    ) -> Self {
        Self::from_vec(slice.to_vec(), shape, device)
    }

    pub fn into_vec<T: DataTypeConversion>(self) -> Vec<T> {
        self.make_concrete();
        let inner = self.inner.lock();
        let Data::Concrete(data) = &inner.data else {
            unreachable!("make_concrete() was called")
        };
        match data {
            #[cfg(feature = "cuda")]
            ConcreteData::Cuda(tensor) => cuda::tensor_to_vec(tensor),
            #[cfg(feature = "cpu")]
            ConcreteData::Cpu => todo!(),
        }
    }

    /// # Panics
    /// Panics if the tensor does not have a length of exactly 1.
    pub fn into_scalar<T: DataTypeConversion>(self) -> T {
        let vec = self.into_vec::<T>();
        assert_eq!(
            vec.len(),
            1,
            "Tensor::into_scalar called on tensor of length != 1"
        );
        vec[0]
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn shape(&self) -> [usize; D] {
        self.inner
            .lock()
            .data
            .shape()
            .dims()
            .try_into()
            .expect("dimension does not match D const parameter?")
    }

    pub fn data_type(&self) -> DataType {
        self.inner.lock().data_type()
    }

    pub fn recip(self) -> Self {
        self.op_unary_pointwise(UnaryPointwiseOp::Recip)
    }

    /// Column-major matrix multiplication.
    pub fn matmul(self, rhs: Self) -> Self {
        const {
            if D < 2 {
                panic!("matrix multiplication requires tensors of dimension >= 2");
            }
        }

        let shape_lhs = self.shape();
        let shape_rhs = rhs.shape();

        assert_eq!(
            shape_lhs[D - 2],
            shape_rhs[D - 1],
            "invalid dimensions for matmul: {}x{} (lhs) is not compatible with {}x{} (rhs)",
            shape_lhs[D - 1],
            shape_lhs[D - 2],
            shape_rhs[D - 1],
            shape_rhs[D - 2],
        );

        assert_eq!(
            self.data_type(),
            rhs.data_type(),
            "matmul only supported when A and B have the same data type"
        );

        if D > 2 {
            assert_eq!(
                &self.shape()[..D - 2],
                &rhs.shape()[..D - 2],
                "for batched matmul, all batch dimensions must have the same size"
            );
        }

        let (graph, lhs) = self.make_graph();
        let rhs = rhs.to_graph(&graph);
        Self::from_op(
            &graph,
            Op::Matmul(op::Matmul {
                input_a: lhs,
                input_b: rhs,
            }),
        )
    }

    /// Swaps last two dimensions.
    pub fn transpose(self) -> Self {
        const {
            if D < 2 {
                panic!("transpose requires at least two dimensions");
            }
        }

        let (cx, this) = self.make_graph();
        Self::from_op(&cx, Op::Transpose(op::Transpose { input: this }))
    }

    /// Broadcasts the given dimension, which must be of size 1.
    pub fn broadcast(self, axis: i32, new_size: usize) -> Self {
        let (cx, this) = self.make_graph();

        let axis = Self::axis_to_unsigned(axis);

        assert!(new_size > 0, "must broadcast to a positive size");
        assert_eq!(
            self.shape()[axis],
            1,
            "can only broadcast an axis of size 1"
        );

        Self::from_op(
            &cx,
            Op::Restructure(op::Restructure {
                input: this,
                op: RestrctureOp::BroadcastAxis {
                    axis: BroadcastAxis::Existing(axis),
                    new_size,
                },
            }),
        )
    }

    /// Expands the tensor size by 1 dimension.
    pub fn expand<const D2: usize>(self, new_axis_size: usize) -> Tensor<D2> {
        const {
            if D2 != D + 1 {
                panic!("output tensor must have exactly one additional dimension");
            }
        }

        assert!(new_axis_size > 0, "must broadcast to a positive size");

        let (cx, this) = self.make_graph();

        Tensor::from_op(
            &cx,
            Op::Restructure(op::Restructure {
                input: this,
                op: RestrctureOp::BroadcastAxis {
                    axis: BroadcastAxis::Expand,
                    new_size: new_axis_size,
                },
            }),
        )
    }

    /// Converts Python-style dimension indexes to usize.
    const fn axis_to_unsigned(i: i32) -> usize {
        let result = if i < 0 {
            D - ((-i) as usize)
        } else {
            i as usize
        };
        if result >= D {
            panic!("axis index out of bounds");
        }
        result
    }

    pub fn pow_scalar(self, power: f32) -> Self {
        self.op_unary_pointwise_scalar(UnaryPointwiseOp::PowScalar, power)
    }

    /// Performs sum reduction along the last `depth` dimensions
    /// of the tensor. The last `depth` dimensions are replaced
    /// with a single dimension of length 1.
    pub fn reduce_sum<const D2: usize>(self, depth: u32) -> Tensor<D2> {
        self.op_reduce(ReduceOp::Sum, depth)
    }

    /// Performs mean reduction along the last `depth` dimensions
    /// of the tensor. The last `depth` dimensions are replaced
    /// with a single dimension of length 1.
    pub fn reduce_mean<const D2: usize>(self, depth: u32) -> Tensor<D2> {
        self.op_reduce(ReduceOp::Mean, depth)
    }

    /// Performs min reduction along the last `depth` dimensions
    /// of the tensor. The last `depth` dimensions are replaced
    /// with a single dimension of length 1.
    pub fn reduce_min<const D2: usize>(self, depth: u32) -> Tensor<D2> {
        self.op_reduce(ReduceOp::Min, depth)
    }

    /// Performs max reduction along the last `depth` dimensions
    /// of the tensor. The last `depth` dimensions are replaced
    /// with a single dimension of length 1.
    pub fn reduce_max<const D2: usize>(self, depth: u32) -> Tensor<D2> {
        self.op_reduce(ReduceOp::Max, depth)
    }

    pub fn reshape<const D2: usize>(self, new_shape: [usize; D2]) -> Tensor<D2> {
        let new_shape = Shape::new(new_shape);
        assert_eq!(
            self.shape().iter().product::<usize>(),
            new_shape.num_elements(),
            "reshape() called with non-matching shape"
        );

        let (cx, this) = self.make_graph();
        Tensor::from_op(
            &cx,
            Op::Reshape(op::Reshape {
                input: this,
                new_shape,
            }),
        )
    }

    /// Verifies that `D2 == D`, and returns `self` as a `Tensor<D2>`.
    /// This is used to bypass some type system limitations.
    pub fn transmute_dim<const D2: usize>(self) -> Tensor<D2> {
        const {
            if D != D2 {
                panic!("D != D2 for transmute_dim()");
            }
        }
        Tensor {
            device: self.device,
            inner: self.inner,
        }
    }
}

impl Tensor<1> {
    pub fn from_scalar<T: DataTypeConversion>(x: T, device: Device) -> Self {
        Self::from_vec(vec![x], [1], device)
    }

    pub fn from_array<T: DataTypeConversion, const N: usize>(arr: [T; N], device: Device) -> Self {
        Self::from_vec(arr.to_vec(), [N], device)
    }
}

impl Tensor<2> {
    pub fn from_array<T: DataTypeConversion, const N1: usize, const N2: usize>(
        arr: [[T; N2]; N1],
        device: Device,
    ) -> Self {
        Self::from_vec(
            arr.into_iter().flatten().collect::<Vec<T>>(),
            [N1, N2],
            device,
        )
    }
}

impl Tensor<3> {
    pub fn from_array<T: DataTypeConversion, const N1: usize, const N2: usize, const N3: usize>(
        arr: [[[T; N3]; N2]; N1],
        device: Device,
    ) -> Self {
        Self::from_vec(
            arr.into_iter().flatten().flatten().collect::<Vec<T>>(),
            [N1, N2, N3],
            device,
        )
    }
}

impl Tensor<4> {
    pub fn from_array<
        T: DataTypeConversion,
        const N1: usize,
        const N2: usize,
        const N3: usize,
        const N4: usize,
    >(
        arr: [[[[T; N4]; N3]; N2]; N1],
        device: Device,
    ) -> Self {
        Self::from_vec(
            arr.into_iter()
                .flatten()
                .flatten()
                .flatten()
                .collect::<Vec<T>>(),
            [N1, N2, N3, N4],
            device,
        )
    }
}

impl<const D: usize> Neg for Tensor<D> {
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        self.op_unary_pointwise(UnaryPointwiseOp::Neg)
    }
}

impl<const D: usize> Add for Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: Self) -> Self::Output {
        self.op_binary_pointwise(&rhs, BinaryPointwiseOp::Add)
    }
}

impl<const D: usize> Add<f32> for Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: f32) -> Self::Output {
        self.op_unary_pointwise_scalar(UnaryPointwiseOp::AddScalar, rhs)
    }
}

impl<const D: usize> Sub for Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl<const D: usize> Sub<f32> for Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: f32) -> Self::Output {
        self + -rhs
    }
}

impl<const D: usize> Mul for Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.op_binary_pointwise(&rhs, BinaryPointwiseOp::Mul)
    }
}
impl<const D: usize> Mul<f32> for Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: f32) -> Self::Output {
        self.op_unary_pointwise_scalar(UnaryPointwiseOp::MulScalar, rhs)
    }
}

impl<const D: usize> Div for Tensor<D> {
    type Output = Tensor<D>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.recip()
    }
}

impl<const D: usize> Div<f32> for Tensor<D> {
    type Output = Tensor<D>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: f32) -> Self::Output {
        self * rhs.recip()
    }
}

/// Internal functions.
impl<const D: usize> Tensor<D> {
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

    /// Moves the tensor onto the given graph. If the tensor
    /// is concrete, this registers the tensor as an input
    /// to the graph. If the tensor is virtual, then we merge
    /// the two graphs if they differ.
    fn to_graph(&self, builder: &OpGraphBuilderHandle) -> NodeId {
        let guard = self.inner.lock();
        match &guard.data {
            Data::Virtual(virt) => {
                if Arc::ptr_eq(&virt.graph, builder) {
                    virt.node
                } else {
                    let cx2 = virt.graph.clone();
                    drop(guard);
                    cx2.lock().merge_into(builder);
                    let guard = self.inner.lock();
                    let Data::Virtual(virt) = &guard.data else {
                        unreachable!()
                    };
                    virt.node
                }
            }
            Data::Concrete(_) => {
                builder
                    .lock()
                    .new_input(guard.shape(), guard.data_type(), &self.inner)
            }
        }
    }

    fn op_binary_pointwise(&self, rhs: &Self, op: BinaryPointwiseOp) -> Self {
        let (cx, this) = self.make_graph();
        Tensor::from_op(
            &cx,
            Op::BinaryPointwise(BinaryPointwise {
                lhs: this,
                rhs: rhs.to_graph(&cx),
                op,
            }),
        )
    }

    fn op_unary_pointwise(&self, op: UnaryPointwiseOp) -> Self {
        let (cx, this) = self.make_graph();
        Tensor::from_op(&cx, Op::UnaryPointwise(UnaryPointwise { input: this, op }))
    }

    fn op_unary_pointwise_scalar(
        &self,
        op: impl Fn(VarId) -> UnaryPointwiseOp,
        value: f32,
    ) -> Self {
        let (cx, this) = self.make_graph();
        let var = VarId::new();
        cx.lock().vars.insert(var, Var::Scalar(value));
        Tensor::from_op(
            &cx,
            Op::UnaryPointwise(UnaryPointwise {
                input: this,
                op: op(var),
            }),
        )
    }

    fn op_reduce<const D2: usize>(&self, op: ReduceOp, depth: u32) -> Tensor<D2> {
        assert_eq!(
            D2,
            D - depth as usize + 1,
            "result tensor dimensions do not match reduction depth"
        );

        let (cx, this) = self.make_graph();
        Tensor::from_op(
            &cx,
            Op::Reduce(Reduce {
                input: this,
                op,
                depth,
            }),
        )
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
            Data::Concrete(conc) => match conc {
                #[cfg(feature = "cuda")]
                ConcreteData::Cuda(tensor) => tensor.shape().clone(),
                #[cfg(feature = "cpu")]
                ConcreteData::Cpu => todo!(),
            },
        }
    }

    pub fn data_type(&self) -> DataType {
        match &self.data {
            Data::Virtual(virt) => virt.data_type,
            Data::Concrete(conc) => match conc {
                #[cfg(feature = "cuda")]
                ConcreteData::Cuda(tensor) => tensor.data_type(),
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
    Cuda(crate::cuda::tensor::RawTensor),
    #[cfg(feature = "cpu")]
    #[expect(unused)]
    Cpu,
}

impl ConcreteData {
    fn shape(&self) -> &Shape {
        match self {
            ConcreteData::Cuda(x) => x.shape(),
            ConcreteData::Cpu => todo!(),
        }
    }
}

type OpGraphBuilderHandle = Arc<Mutex<OpGraphBuilder>>;

struct OpGraphBuilder {
    device: Device,
    op_graph: OpGraph,
    inputs: SecondaryMap<NodeId, Arc<Mutex<TensorInner>>>,
    node_to_tensor: SecondaryMap<NodeId, Weak<Mutex<TensorInner>>>,
    vars: VarMap,
}

impl OpGraphBuilder {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            op_graph: OpGraph::new(),
            inputs: SecondaryMap::default(),
            node_to_tensor: SecondaryMap::default(),
            vars: VarMap::new(),
        }
    }

    pub fn merge_into(&mut self, other_handle: &OpGraphBuilderHandle) {
        let mut other = other_handle.lock();
        let node_mapping = self.op_graph.merge_into(&mut other.op_graph);

        for (old_node_id, tensor) in &self.node_to_tensor {
            let new_node_id = node_mapping[old_node_id];
            if let Some(tensor) = tensor.upgrade() {
                let mut guard = tensor.lock();
                if let Data::Virtual(virt) = &mut guard.data {
                    virt.graph = Arc::clone(other_handle);
                    virt.node = new_node_id;
                }
                other
                    .node_to_tensor
                    .insert(new_node_id, Arc::downgrade(&tensor));
            }
        }

        for (old_input_id, tensor) in &self.inputs {
            let new_input_id = node_mapping[old_input_id];
            other.inputs.insert(new_input_id, Arc::clone(tensor));
        }

        for (var, value) in self.vars.drain() {
            other.vars.insert(var, value);
        }
    }

    pub fn new_input(
        &mut self,
        shape: Shape,
        data_type: DataType,
        tensor: &Arc<Mutex<TensorInner>>,
    ) -> NodeId {
        let id = self.op_graph.new_input(Descriptor { shape, data_type });
        self.inputs.insert(id, Arc::clone(tensor));
        self.node_to_tensor.insert(id, Arc::downgrade(tensor));
        id
    }

    pub fn new_op(&mut self, op: Op, owner: &Arc<Mutex<TensorInner>>) -> NodeId {
        let node = self.op_graph.new_op(op);
        self.node_to_tensor.insert(node, Arc::downgrade(owner));
        node
    }

    pub fn resolve(&mut self) {
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
            Device::Cuda(device_index) => cuda::resolve(self, *device_index),
            #[cfg(feature = "cpu")]
            Device::Cpu => todo!(),
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::cuda::{context::CudaContext, execution::TensorMap, tensor::RawTensor};

    pub fn resolve(graph: &mut OpGraphBuilder, device_index: u32) {
        let cx = CudaContext::global(device_index).expect("failed to create CUDA context");
        let plan = crate::cuda::planner::compile_plan(cx, &Arc::new(graph.op_graph.clone()))
            .expect("failed to generate plan");

        let mut inputs = TensorMap::new();
        for &input in graph.op_graph.inputs() {
            let tensor = graph.inputs[input].lock();
            let Data::Concrete(data) = &tensor.data else {
                panic!("input tensor cannot be virtual")
            };

            // TODO: check device / handle multi-device
            let ConcreteData::Cuda(data) = &data else {
                panic!("data must reside on CUDA")
            };

            inputs.insert(input, data.clone());
        }

        let mut outputs = crate::cuda::execution::execute_plan(&plan, cx, inputs, &graph.vars)
            .expect("failed to execute plan");

        for (node, tensor) in &graph.node_to_tensor {
            if let Some(tensor) = tensor.upgrade() {
                let val = outputs.remove(node).unwrap();
                tensor.lock().data = Data::Concrete(ConcreteData::Cuda(val));
            }
        }
    }

    pub fn tensor_to_vec<T: DataTypeConversion>(tensor: &RawTensor) -> Vec<T> {
        tensor
            .to_vec_sync()
            .expect("failed to transfer data to host")
    }
}
